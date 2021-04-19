# Source code

## To start the benchmark run:
1. `docker build -t bench_img .`
2. `docker run -dit --name bench_1 bench_img`
3. `docker exec -it bench_1 bash`
4. `python bench_paper_solo_all.py`

