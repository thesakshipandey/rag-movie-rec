.PHONY: router-xgb-train router-xgb-eval

router-xgb-train:
	python -m src.router.train_router_xgb \
		--features_parquet artifacts/router/features.parquet \
		--out_dir artifacts/router/xgb \
		--config configs/router_xgb.yaml \
		--seed 42

router-xgb-eval:
	python -m src.router.eval_router_xgb \
		--features_parquet artifacts/router/features.parquet \
		--models_dir artifacts/router/xgb \
		--out_dir artifacts/router/xgb_eval
