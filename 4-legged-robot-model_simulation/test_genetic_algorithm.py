import time
import pandas as pd
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False



N = 25  # grid dimension
device = 'cuda'
TEST_CSV = '../input/conways-reverse-game-of-life-2020/test.csv'
OUTPUT_CSV = 'submission.csv'





cv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='circular', bias=False)
cv.requires_grad=False
cv.weight = torch.nn.Parameter(
    torch.tensor(
        [[[[ 1., 1., 1.],
           [ 1., 0., 1.],
           [ 1., 1., 1.]]]],
        device=device,
        dtype=torch.float16
    ),
    requires_grad=False,
)


@torch.jit.script
def forward(grid, delta: int):
    N=25
    g = grid.reshape(-1, 1, N, N)
    for _ in torch.arange(delta):
        g = g.to(torch.float16)
        neighbor_sum = cv(g)
        g = ((neighbor_sum == 3) | ((g == 1) & (neighbor_sum == 2)))
    return g.reshape(-1, N, N)

@torch.jit.script
def random_parents(n_parents: int, device: str):
    N = 25
    RANDOM_ALIVE = .2
    return torch.rand((n_parents, N, N), device=device) > (1-RANDOM_ALIVE)


@torch.jit.script
def loss(input, target):
    return torch.sum(input ^ target, dim=(-1,-2))


@torch.jit.script
def select_best(parents, delta: int, target, n_best: int):
    scores = loss(forward(parents, delta), target)
    best_values, best_indices = torch.topk(scores, n_best, dim=0, largest=False, sorted=True)
    new_parents = parents[best_indices, ...]
    return new_parents, best_values[0], new_parents[0, ...]


@torch.jit.script
def random_combine(parents, n_offsprings: int, device: str, pre_masks):
    N = 25
    
    dads = torch.randint(low=0, high=parents.shape[0], size=(n_offsprings,),
                         device=device, dtype=torch.long)
    dads = parents[dads, ...]
    
    moms = torch.randint(low=0, high=parents.shape[0], size=(n_offsprings,),
                         device=device, dtype=torch.long)
    moms = parents[moms, ...]
    
    masks = pre_masks[torch.randint(low=0, high=pre_masks.shape[0], size=(n_offsprings,),
                                    device=device, dtype=torch.long)]

    return torch.where(masks, dads, moms)



def precomputes_masks():
    N = 25
    BLOCK_SIZE = 17

    block = torch.nn.Conv2d(1, 1, kernel_size=BLOCK_SIZE, padding=BLOCK_SIZE//2,
                            padding_mode='circular', bias=False)
    block.requires_grad=False
    block.weight = torch.nn.Parameter(
        torch.ones((1, 1, BLOCK_SIZE, BLOCK_SIZE),
            device=device,
            dtype=torch.float16
        ),
        requires_grad=False,
    )

    masks = torch.zeros((N * N, 1, N, N), device=device, dtype=torch.float16)
    
    for x in range(N):
        for y in range(N):
            masks[x * N + y, 0, x, y] = 1.
    masks = block(masks)
    
    return masks[:, 0, ...] > .5

@torch.jit.script
def mutate(parents, device: str):
    MUTATION = .0016  # .005 
    mutations = torch.rand(parents.shape, device=device) < MUTATION
    return parents ^ mutations

@torch.jit.script
def optimize_one_puzzle(delta: int, data, device: str, pre_masks):
    N = 25
    N_GENERATION = 30  # Number of generations
    P = 4_500  # population
    N_BEST = P // 30  # best to keep as new parents
    N_ELITES = 8  # parents unchanged for next generation
    
    best_score = torch.tensor([N*N], device=device)
    best = torch.zeros((N,N), device=device).to(torch.bool)
    parents = random_parents(P, device)

    elites = torch.empty((1, N, N), dtype=torch.bool, device=device)
    elites[0, ...] = data  # set target as potential dad ;)

    for i in range(N_GENERATION):
        parents = random_combine(parents, P, device, pre_masks)
        parents = mutate(parents, device)
        parents[:N_ELITES, ...] = elites
        parents, best_score, best = select_best(parents, delta, data, N_BEST)
        # Some of the individuals in the current population that have lower fitness are chosen as elite.
        # These elite individuals are passed to the next population.
        elites = parents[:N_ELITES, ...]
        if best_score == 0:  # early stopping
            break

    return best_score, best

@torch.jit.script
def optimize_all_puzzles(deltas, df, device: str, pre_masks):
    sub = df.clone()
    
    for n in torch.arange(df.shape[0]):
        delta = deltas[n]
        data = df[n, ...]
        _, sub[n, ...] = optimize_one_puzzle(delta, data, device, pre_masks)

    return sub


df = pd.read_csv(TEST_CSV, index_col='id')


submission = df.copy()
submission.drop(['delta'], inplace=True, axis=1)



indexes = df.index
deltas = torch.from_numpy(df.delta.values).to(device)
df = torch.BoolTensor(df.values[:, 1:].reshape((-1, N, N))).to(device)

start_time = time.time()
pre_masks = precomputes_masks()
sub = optimize_all_puzzles(deltas, df, device, pre_masks)
print(f'Processed {sub.shape[0]:,} puzzles in {time.time() - start_time:.2f} seconds 🔥🔥🔥')



submission.rename(columns={f'stop_{x}': f'start_{x}' for x in range(N*N)}, inplace=True)
submission.iloc[:sub.shape[0], :] = sub.reshape((-1, N*N)).cpu().numpy().astype(int)
submission.to_csv(OUTPUT_CSV)



def leaderboard_score(deltas, df, sub, device: str):
    result = torch.empty(sub.shape[0], device=device, dtype=torch.long)
    for delta in range(1, 6):
        start = sub[deltas == delta]
        end   = df[deltas == delta]
        result[deltas == delta] = loss(forward(start, delta), end)
    print('Leaderboard score:', torch.sum(result).item() / (result.shape[0]*N*N))



leaderboard_score(deltas, df, sub, device)



"""
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
"""