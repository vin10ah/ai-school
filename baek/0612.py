# from collections import defaultdict

# v, e = map(int, input().split())
# k = int(input())
# graph = defaultdict(dict)


# for _ in range(e):
#     u,v,w = map(int, input().split())
#     graph[u][v]=w

# dists = {node:float('inf') for node in range(1, e+1)}
# dists[k] = 0
# q = [(0, k)]
# print(graph)
# while q:
#     q.sort(reverse=True)
#     dist, node = q.pop()
#     print(f'#0, dist:{dist}, node:{node}') #0

#     if dist > dists[node]:
#         print(f'#1 dist:{dist}, dists[{node}]:{dists[node]}') #1
#         continue

#     for n_node, n_dist in graph[node].items():
#         print(f'#2, n_node:{n_node}, n_dist:{n_dist}') #2
#         n_dist += dist
#         print(f'#3, n_dist:{n_dist}') #3

#         if n_dist < dists[n_node]:
#             q.append((n_dist, n_node))
#             dists[n_node] = n_dist
#             print(f'dists[n_node]: {dists[n_node]}')

# if float('inf') in dists.values():
#     print('INF')
# else:
#     print(max(dists.values()))

from collections import defaultdict

n, e = map(int, input().split())
k = int(input())
graph = defaultdict(dict)

for _ in range(e):
    u,v,w = map(int, input().split())
    graph[u][v]=w

dists = {node:float('inf') for node in range(1, n+1)}
dists[k] = 0
q = [(0, k)]

while q:
    q.sort(reverse=True)
    dist, node = q.pop()
    print(dists[node])
    
    if dist > dists[node]:
        continue

    for n_node, n_dist in graph[node].items():
        n_dist += dist

        if n_dist < dists[n_node]:
            q.append((n_dist, n_node))
            dists[n_node] = n_dist
    
    

# for i in range(1,n+1):
#     print(str(dists[i]).upper())            

    # if dists[node] == 'inf':
    #     print('INF')
    # else:
    #     print(dists[node])
# print(dists)
# for dist in dists.values():
#     print(dist)
# # print(dists.values())