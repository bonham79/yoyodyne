yoyodyne-train --model_dir=../ \
--experiment='new' \
--train='../2024G2PST/data/tsv/latin/latin_train_label.tsv' \
--val='../2024G2PST/data/tsv/latin/latin_dev_label.tsv' \
--arch='hmm_lstm' \
--log_every_n_step=10 \
--source_encoder_arch 'lstm' \
--features_encoder_arch='linear' \
--source_attention_heads=4 \
--embedding_size=256 \
--encoder_layers=4 \
--decoder_layers=1 \
--hidden_size=512 \
--batch_size=32 \
--dropout=0.4 \
--learning_rate=0.001 \
--optimizer="adam" \
--max_steps=10000 \
--enforce_monotonic \
--features_col 0 \
--precision=16 \
--attention_context=8 \
--check_val_every_n_epoch 1 \
--accelerator gpu \
--num_sanity_val_steps=2 \
--gradient_clip_val 5 \
--source_coverage=1.0 \
--seed=42