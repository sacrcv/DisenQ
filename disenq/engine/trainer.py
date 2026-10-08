"""Training loop."""

import json
import os
import time
from typing import Dict, Optional

import torch

from disenq.engine.optim import build_optimizer, build_scheduler
from disenq.evaluation import evaluate
from disenq.utils.dist import is_main_process
from disenq.utils.logger import MetricLogger, get_logger
from disenq.utils.misc import load_checkpoint, save_checkpoint, unwrap_model

_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None, "fp32": None}


def resolve_amp_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    key = str(name).lower()
    if key not in _AMP_DTYPES:
        raise ValueError(f"run.amp must be one of {sorted(_AMP_DTYPES)}, got '{name}'")
    return _AMP_DTYPES[key]


class Trainer:
    """Drives training, periodic evaluation and checkpointing."""

    def __init__(
        self,
        cfg,
        model,
        train_loader,
        device: torch.device,
        query_loader=None,
        gallery_loader=None,
        output_dir: str = "output",
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.device = device
        self.output_dir = output_dir
        self.logger = get_logger()

        run_cfg = cfg.run
        self.max_epoch = run_cfg.get("max_epoch", 60)
        self.accum_grad_iters = max(run_cfg.get("accum_grad_iters", 1), 1)
        self.clip_grad_norm = run_cfg.get("clip_grad_norm", 5.0)
        self.print_freq = run_cfg.get("print_freq", 20)
        self.eval_freq = run_cfg.get("eval_freq", 10)
        self.save_freq = run_cfg.get("save_freq", 10)
        self.protocols = list(run_cfg.get("protocols", ["same_activity"]))
        self.primary_protocol = run_cfg.get("primary_protocol", self.protocols[0])
        self.primary_metric = run_cfg.get("primary_metric", "R@1")

        self.amp_dtype = resolve_amp_dtype(run_cfg.get("amp", "bf16"))
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16 and device.type == "cuda")
        )

        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer, len(train_loader))

        self.start_epoch = 0
        self.best_score = -1.0
        self.history = []

    # ------------------------------------------------------------------
    def resume(self, path: str) -> None:
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.scaler, strict=False
        )
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        best = ckpt.get("best")
        if isinstance(best, (int, float)):
            self.best_score = float(best)
        self.logger.info(f"resuming at epoch {self.start_epoch} (best={self.best_score:.2f})")

    # ------------------------------------------------------------------
    def _move_batch(self, batch: Dict) -> Dict:
        out = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device, non_blocking=True)
            else:
                out[key] = value
        return out

    def _autocast(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        metric_logger = MetricLogger()
        header = f"epoch {epoch}/{self.max_epoch - 1}"
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in metric_logger.log_every(
            self.train_loader, self.print_freq, header=header
        ):
            batch = self._move_batch(batch)

            with self._autocast():
                output = self.model(batch)
                loss = output["loss"] / self.accum_grad_iters

            if not torch.isfinite(loss):
                self.logger.warning(f"non-finite loss at epoch {epoch} step {step}; skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_grad_iters == 0:
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            metric_logger.update(
                loss=output["loss"],
                l_id=output["loss_id"],
                l_tri=output["loss_tri"],
                l_orth=output["loss_orth"],
                l_act=output["loss_act"],
                l_sim=output["loss_sim"],
                id_acc=output["id_acc"],
                act_acc=output["act_acc"],
                alpha_b=output["alpha_biometrics"],
                lr=self.scheduler.get_last_lr()[0],
            )

        stats = metric_logger.global_avg_dict()
        self.logger.info(f"{header} summary: " + "  ".join(
            f"{k}={v:.4f}" for k, v in stats.items()
        ))
        return stats

    # ------------------------------------------------------------------
    def can_evaluate(self) -> bool:
        return self.query_loader is not None and self.gallery_loader is not None

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        run_cfg = self.cfg.run
        return evaluate(
            self.model,
            self.query_loader,
            self.gallery_loader,
            self.device,
            protocols=self.protocols,
            amp_dtype=self.amp_dtype,
            use_motion=run_cfg.get("use_motion", True),
            adaptive=run_cfg.get("adaptive_similarity", True),
            fixed_alpha=run_cfg.get("fixed_alpha", 0.5),
            max_rank=run_cfg.get("max_rank", 50),
            print_freq=self.print_freq,
            feature=run_cfg.get("eval_feature", "fused"),
        )

    def _score(self, results: Dict[str, Dict[str, float]]) -> float:
        block = results.get(self.primary_protocol)
        if block is None:
            block = next(iter(results.values()))
        return float(block.get(self.primary_metric, block.get("R@1", 0.0)))

    @staticmethod
    def _dump_history(path: str, history) -> None:
        if not is_main_process():
            return
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        start = time.time()
        for epoch in range(self.start_epoch, self.max_epoch):
            stats = self.train_one_epoch(epoch)
            entry = {"epoch": epoch, "train": stats}

            is_last = epoch == self.max_epoch - 1
            should_eval = self.can_evaluate() and (
                is_last or (self.eval_freq > 0 and (epoch + 1) % self.eval_freq == 0)
            )
            if should_eval:
                results = self.evaluate()
                entry["eval"] = results
                score = self._score(results)
                if score > self.best_score:
                    self.best_score = score
                    save_checkpoint(
                        os.path.join(self.output_dir, "checkpoint_best.pth"),
                        self.model, self.optimizer, self.scheduler, self.scaler,
                        epoch=epoch, best=self.best_score, cfg=self.cfg,
                    )
                    self.logger.info(
                        f"new best {self.primary_protocol}/{self.primary_metric} = "
                        f"{self.best_score:.2f}"
                    )

            if is_last or (self.save_freq > 0 and (epoch + 1) % self.save_freq == 0):
                save_checkpoint(
                    os.path.join(self.output_dir, "checkpoint_last.pth"),
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch=epoch, best=self.best_score, cfg=self.cfg,
                )

            self.history.append(entry)
            self._dump_history(os.path.join(self.output_dir, "history.json"), self.history)

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        self.logger.info(f"training finished in {elapsed}; best={self.best_score:.2f}")
        return {"best": self.best_score, "history": self.history}
