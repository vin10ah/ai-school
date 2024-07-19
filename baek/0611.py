import sys

sys.stdin = open('0611.txt') # 표준 입력값 지정

while True: 
    w, h = map(int, input().split())
    map_ = []
    cnt = 0

    if w == h == 0:
        break

    else:
        for i in range(h):
            map_.append(list(map(int, input().split()))) 






def dfs(now):
    x, y = now
    map_[x][y] = 0
    for nx in (x+1,x,x-1):
        for ny in (y+1, y, y-1):
            if 0 <= nx < h and 0 <= ny < w and map_[nx][ny]==1:
                dfs((nx, ny))


while True:
    w, h = map(int, input().split())
    map_ = []
    cnt = 0

    if w == h == 0:
        break

    else:
        for i in range(h):
            map_.append(list(map(int, input().split())))

    for idx, i in enumerate(map_):
        for jdx, j in enumerate(i):
            if j == 1:
                dfs((idx, jdx))
                cnt += 1
    
    print(cnt)

    # for i in range(h):
    #     cnt = 0
    #     for j in range(3):
    #         if i == 0 and j==2:
    #             map_[i][j]

    # print(*map_, sep='\n')
    # print()


