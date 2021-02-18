python metrics_estimation.py --n-betas=50 --n-points-per-beta=5000  --beta-min="1e-12" --beta-max=53.0 --method="bootstrap" <output-dir>/1M/amazing/direct-amazing.pkl
python metrics_estimation.py --n-betas=50 --n-points-per-beta=5000  --beta-min="1e-12" --beta-max=6.0 --method="bootstrap" <output-dir>/1M/wikileaks/direct-wikileaks.pkl
python metrics_estimation.py --n-betas=50 --n-points-per-beta=5000  --beta-min="4e-7" --beta-max="4e3" --method="bootstrap" <output-dir>/1M/female/direct-female.pkl
python metrics_estimation.py --n-betas=50 --n-points-per-beta=5000  --beta-min="1e-12" --beta-max="9.3e6" --method="bootstrap" <output-dir>/1M/female_science/direct-female_science.pkl
python metrics_estimation.py --n-betas=50 --n-points-per-beta=5000  --beta-min="1e-12" --beta-max="2.9e7" --method="bootstrap" <output-dir>/1M/female_sports/direct-female_sports.pkl
