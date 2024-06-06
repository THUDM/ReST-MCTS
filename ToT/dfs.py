from ToT.base import Node, rand_select


def DFS_sub(tot_task, node):
    if node.depth >= tot_task.max_depth:
        print('达到最大深度限制!\n')
        return "", node, None

    candidates = []
    for i in range(tot_task.branch):
        new_pcd = ''
        cnt = 3
        while not new_pcd and cnt:
            new_pcd = tot_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not new_pcd:
            continue

        node, child = node.append_children(new_pcd)
        value = tot_task.get_step_value(child.y)
        child.update_value(value)
        child.visit_sequence = tot_task.node_count
        tot_task.update_count()
        candidates.append(child)

    if not candidates:
        print('未找到合适的下一步!\n')
        return "", node, None
    ranked_candidates = sorted(candidates, key=lambda item: item.V, reverse=True)
    if ranked_candidates[0].V >= tot_task.end_gate:
        ranked_candidates[0].final_ans_flag = 1
        return ranked_candidates[0].y, node, ranked_candidates[0]

    # 继续下探
    if tot_task.select_method == 'greedy':
        selected = ranked_candidates[:min(tot_task.select_branch, tot_task.branch, len(ranked_candidates))]

    else:
        idx_list = []
        selected = []
        for j in range(min(tot_task.select_branch, tot_task.branch)):
            idx, node = rand_select(ranked_candidates, [item.V for item in ranked_candidates])
            if idx not in idx_list:
                idx_list.append(idx)
                selected.append(node)
        selected = sorted(selected, key=lambda item: item.V, reverse=True)

    for child in selected:
        solution, child, final_node = DFS_sub(tot_task, child)
        if solution:
            return solution, node, final_node

    return "", node, None


def DFS(tot_task):
    root = Node('')
    solution, root, final_node = DFS_sub(tot_task, root)
    if solution:
        print(f'已找到最终解!\nSolution:{solution}\n')
        return solution, root, final_node
    else:
        max_node, max_V = root.getBestV()
        max_node.final_ans_flag = 1
        print(f'未找到满足要求价值的解答，采用最高价值价值解答代替。\nSolution:{max_node.y}\n')
        return max_node.y, root, max_node
