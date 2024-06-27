import logging
import math
import os
import random
import sys
from typing import Tuple
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.utils.data.distributed

from src.ir import Retriever, RetrieverConfig
from src.ir.utils.biencoder_utils import create_biencoder_batch
from src.ir.data.biencoder_dataset import BiencoderDatasetsCfg
from src.ir.data.ddp_iterators import MultiSetDataIterator, get_data_iterator
from src.ir.training.conf_utils import setup_cfg_gpu, set_seed, setup_logger
from src.ir.training.model_utils import get_optimizer, get_schedule_linear, CheckpointState
from src.ir.training.loss_utils import _do_biencoder_fwd_pass
from src.ir.training.ddp_utils import is_master

logger = logging.getLogger()
setup_logger(logger)

class RetrieverTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1
        self.ds_cfg = BiencoderDatasetsCfg(cfg)
        self.init_retriever()        
        self.scaler = torch.cuda.amp.GradScaler()
        self.start_epoch = 0
        self.start_batch = 0

    def init_retriever(self):
        logger.info("***** Initializing components for training *****")
        if self.cfg.model_path:
            logger.info(f"***** Loading checkpoint from {self.cfg.model_path} *****")
            model = Retriever.from_pretrained(self.cfg.model_path)
        else:
            biencoder_cfg = OmegaConf.to_container(self.cfg.biencoder)        
            retriever_cfg = RetrieverConfig(**biencoder_cfg)
            model = Retriever(retriever_cfg)

        if hasattr(self.cfg, "index") and self.cfg.index is not None:
            logger.info("***** Initializing Index: %s", self.cfg.index)
            index = hydra.utils.instantiate(self.cfg.index_stores[self.cfg.index])            
            index.to_cuda(device=self.cfg.device)
            model.index = index

        if is_master():
            logger.debug(f"model.embedding_q.sum: {model.encoder_q.bert_model.embeddings.word_embeddings.weight.sum()}")
            logger.debug(f"model.embedding_p.sum: {model.encoder_p.bert_model.embeddings.word_embeddings.weight.sum()}")

        model.to(self.cfg.device)
        if self.cfg.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if self.cfg.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.cfg.device], output_device=self.cfg.local_rank, find_unused_parameters=True)
        self.model = model
        
        optimizer = get_optimizer(model, learning_rate=self.cfg.train.learning_rate, adam_eps=self.cfg.train.adam_eps, weight_decay=self.cfg.train.weight_decay)
        self.optimizer = optimizer


    def run_train(self):
        cfg = self.cfg
        train_iterator = get_data_iterator(self.ds_cfg, 
                                           cfg.train.batch_size, 
                                           True,
                                           shuffle=True,
                                           shuffle_seed=cfg.seed,
                                           offset=self.start_batch,
                                           rank=cfg.local_rank)
        max_iterations = train_iterator.get_max_iterations()
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = train_iterator.max_iterations
        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        warmup_steps = cfg.train.num_warmup_epochs * updates_per_epoch

        logger.info(" Total updates=%d", total_updates)
        logger.info(" Warmup updates=%d", warmup_steps)
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)
        logger.info("***** Training *****")

        self.save_checkpoint("0")
        for epoch in range(self.start_epoch+1, int(cfg.train.num_train_epochs)+1):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info("***** Training Finished. *****")

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        train_data_iterator: MultiSetDataIterator,
    ):

        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        correct_predictions_1 = 0
        correct_predictions_2 = 0

        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        seed = cfg.seed
        self.model.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0
        
        dataset = 0
        for i, samples_batch in enumerate(train_data_iterator.iterate_ds_data(epoch=epoch)):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            answers = [x.answers for x in samples_batch]
            ds_cfg = self.ds_cfg.train_datasets[dataset]
            shuffle_positives = ds_cfg.shuffle_positives

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)
            
            biencoder_batch = create_biencoder_batch(
                self.model.module,
                samples=samples_batch,
                insert_title=self.cfg.train.train_insert_title,
                num_hard_negatives=cfg.train.hard_negatives,
                num_other_negatives=cfg.train.other_negatives,
                shuffle=True,
                shuffle_positives=shuffle_positives,
            )

            print_flag = (i % log_result_step == 0)

            with autocast():
                loss, is_correct_1, is_correct_2 = _do_biencoder_fwd_pass(
                    cfg,
                    self.model,
                    biencoder_batch,
                    answers=answers,
                    verbose=print_flag,
                    logger=logger,
                )

            correct_predictions_1 += is_correct_1
            correct_predictions_2 += is_correct_2
            global_batch_size = cfg.train.batch_size * cfg.n_gpu * cfg.distributed_world_size
            
            top1_acc_1 = f"{is_correct_1}/{global_batch_size}={round(is_correct_1/global_batch_size*100, 1)}"
            top1_acc_2 = f"{is_correct_2}/{global_batch_size}={round(is_correct_2/global_batch_size*100, 1)}"
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()
            self.scaler.scale(loss).backward()   
            self.scaler.unscale_(self.optimizer)         
            if cfg.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.max_grad_norm)
            scheduler.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.zero_grad()

            if i % log_result_step == 0 and cfg.local_rank in [-1, 0]:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    (f"Epoch: {epoch}: Step: {data_iteration}/{epoch_batches}, "
                     f"loss[v]={loss.item():.4f}, lr={lr:.6f}, "
                     f"acc@1[1]={top1_acc_1}, acc@1[2]={top1_acc_2}")
                )

            if (i + 1) % rolling_loss_step == 0 and cfg.local_rank in [-1, 0]:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

        logger.info("Epoch finished on %d", cfg.local_rank)
        if epoch % cfg.train.num_epoch_to_save == 0:
            _ = self.save_checkpoint(str(epoch))
        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total (1) correct predictions=%d", correct_predictions_1)
        logger.info("epoch total (2) correct predictions=%d", correct_predictions_2)


    def save_checkpoint(self, suffix: str) -> str:
        cfg = self.cfg
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        output_dir = cfg.output_dir if cfg.output_dir else "./"
        cp = os.path.join(output_dir, cfg.save_name_prefix + "_" + suffix)
        model_to_save.save_pretrained(cp, safe_serialization=False) 
        logger.info("Saved checkpoint at %s", cp)
        return cp


@hydra.main(version_base="1.3", config_path="conf", config_name="train_retriever_cfg")
def main(cfg: DictConfig):

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg.output_dir = cfg.output_dir or hydra_cfg.run.dir
    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("Config (after cuda configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = RetrieverTrainer(cfg)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_path and cfg.dev_datasets:
        logger.info("No train files are specified. Run validation only. ")
    else:
        logger.info("Neither train_file or dev_file are specified.")


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
