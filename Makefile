# DraftZone MMM — pipeline targets. See docs/INFRA.md.
.PHONY: data fit experiments anchored evaluate figures dashboard notebooks test all clean

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

dashboard:   ## build the interactive site
	cd dashboard && npm ci && npm run build

notebooks:   ## render notebooks to HTML
	jupyter nbconvert --to html --output-dir docs/notebooks notebooks/*.ipynb

test:
	pytest -q

all: data fit experiments anchored evaluate figures notebooks dashboard

clean:
	rm -rf artifacts/* docs/data/* docs/notebooks/*
