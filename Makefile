# DraftZone MMM — pipeline targets. See docs/INFRA.md.
.PHONY: data fit experiments anchored evaluate figures report ladder leaderboard meridian dashboard notebooks test all clean

data:        ## generate synthetic national + geo data (+ sealed truth)
	python -m draftzone_mmm.datagen

fit:         ## baseline Bayesian fit (multi-core on VM)
	python -m draftzone_mmm.fit_bayes --out artifacts/idata.nc

experiments: ## run rotating geo experiments -> anchors
	python -m draftzone_mmm.experiment --all-channels --out artifacts/anchors.json

anchored:    ## refit with experiment anchors
	python -m draftzone_mmm.fit_bayes --anchors artifacts/anchors.json --out artifacts/idata_anchored.nc

evaluate:    ## grade against sealed truth (ONLY step allowed to read data_sealed/)
	python -m draftzone_mmm.evaluate --out docs/data/scorecard.json

figures:     ## export dashboard data contracts
	python -m draftzone_mmm.export_dashboard_data --out docs/data/

report:      ## publish per-run HTML report + rebuild the runs index (docs/runs/). LABEL=seed77
	python scripts/make_report.py $(if $(LABEL),--label $(LABEL),)

ladder:      ## spend-ladder demo: replica geos + multi-cell ladder -> fit curve -> publish docs/ladder/
	python -m draftzone_mmm.datagen --seed 77 --hetero-geos --spend-ladder
	python -m draftzone_mmm.spend_ladder
	python scripts/spend_ladder_report.py

leaderboard: ## grade every engine (incl. spend ladder) against the sealed truth -> docs/engines/
	python scripts/engine_leaderboard.py

robyn:       ## fit the Robyn-style engine (ridge + Nevergrad + DECOMP.RSSD). Needs nevergrad
	python scripts/fit_robyn_style.py

meridian:    ## fit Google Meridian variants (national Fourier/AKS + geo panel). Needs .[meridian]
	python scripts/fit_meridian.py --mode national --seasonality fourier
	python scripts/fit_meridian.py --mode national --seasonality aks
	python scripts/fit_meridian.py --mode geo
	python scripts/fit_meridian.py --mode geo --demand-control            # imperfect proxy (~0.78)
	python scripts/fit_meridian.py --mode geo --demand-control demand_proxy_hi  # near-perfect (~0.98)

dashboard:   ## build the interactive site
	cd dashboard && npm ci && npm run build

notebooks:   ## render notebooks to HTML
	jupyter nbconvert --to html --output-dir docs/notebooks notebooks/*.ipynb

test:
	pytest -q

all: data fit experiments anchored evaluate figures notebooks dashboard

clean:
	rm -rf artifacts/* docs/data/* docs/notebooks/*
