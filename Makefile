.PHONY: docs-algorithms

docs-algorithms:
	@cd docs/algorithms && pdflatex -interaction=nonstopmode -halt-on-error robust_window_period_and_marker_detection.tex >/dev/null
	@cd docs/algorithms && pdflatex -interaction=nonstopmode -halt-on-error robust_window_period_and_marker_detection.tex >/dev/null
	@echo "Built docs/algorithms/robust_window_period_and_marker_detection.pdf"
