import numpy as np

ACTIONS = ['haut', 'bas', 'gauche', 'droite']

FULL_ACTION = {
    'haut': '^ up',
    'bas': 'v down',
    'gauche': '< left',
    'droite': '> right'
}

SHORT_ACTION = {
    'haut': '^',
    'bas': 'v',
    'gauche': '<',
    'droite': '>'
}

def get_orthogonal_actions(action):
    if action in ['haut', 'bas']:
        return ['gauche', 'droite']
    else:
        return ['haut', 'bas']

def apply_move(state, action, grid, rows, cols):
    r = state // cols
    c = state % cols
    
    nr, nc = r, c
    if action == 'haut': nr -= 1
    elif action == 'bas': nr += 1
    elif action == 'gauche': nc -= 1
    elif action == 'droite': nc += 1
        
    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state
    
    next_state = nr * cols + nc
    if grid[nr][nc] == 3:
        return state
        
    return next_state

def get_reward(state, grid, cols):
    r = state // cols
    c = state % cols
    cell = grid[r][c]
    
    if cell == 1:   return 1.0
    elif cell == 2: return -1.0
    else:           return -0.04

def evaluate_policy(policy, grid, rows, cols, gamma, valid_states):
    N = len(valid_states)
    state_to_idx = {s: i for i, s in enumerate(valid_states)}
    
    A = np.zeros((N, N))
    b = np.zeros(N)
    
    for s in valid_states:
        i = state_to_idx[s]
        A[i, i] = 1.0
        
        action = policy[s]
        intended_next = apply_move(s, action, grid, rows, cols)
        orth_actions = get_orthogonal_actions(action)
        orth1_next = apply_move(s, orth_actions[0], grid, rows, cols)
        orth2_next = apply_move(s, orth_actions[1], grid, rows, cols)
        
        transitions = {}
        transitions[intended_next] = transitions.get(intended_next, 0.0) + 0.8
        transitions[orth1_next] = transitions.get(orth1_next, 0.0) + 0.1
        transitions[orth2_next] = transitions.get(orth2_next, 0.0) + 0.1
        
        expected_reward = 0.0
        for next_s, prob in transitions.items():
            expected_reward += prob * get_reward(next_s, grid, cols)
            if next_s in valid_states:
                j = state_to_idx[next_s]
                A[i, j] -= gamma * prob
                
        b[i] = expected_reward
        
    U_values = np.linalg.solve(A, b)
    U = {s: U_values[state_to_idx[s]] for s in valid_states}
    return U

def print_visualisation(grid, policy, rows, cols, log):
    log.write("---Visualisation---\n\n")
    for r in range(rows):
        row_symbols = []
        for c in range(cols):
            s = r * cols + c
            if grid[r][c] == 3:
                row_symbols.append("o")
            elif grid[r][c] == 1:
                row_symbols.append("B") 
            elif grid[r][c] == 2:
                row_symbols.append("F") 
            else:
                row_symbols.append(SHORT_ACTION[policy[s]])
        log.write("[" + ", ".join(row_symbols) + "]\n")
    log.write("\n")

def solve_policy_iteration(input_filename="policy-iteration.txt", output_filename="log-file_PI.txt"):
    try:
        with open(input_filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Erreur : Le fichier {input_filename} est introuvable.")
        return

    grid = []
    gamma = 0.0
    for line in lines:
        if ',' in line:
            grid.append([int(x) for x in line.split(',')])
        elif gamma == 0.0:
            gamma = float(line)
            
    rows = len(grid)
    cols = len(grid[0])
    num_states = rows * cols
    
    walls = [i for i in range(num_states) if grid[i // cols][i % cols] == 3]
    terminals = [i for i in range(num_states) if grid[i // cols][i % cols] in [1, 2]]
    valid_states = [i for i in range(num_states) if i not in walls and i not in terminals]
    
    initial_policy_map = {
        0: 'haut', 1: 'gauche', 2: 'bas', 
        4: 'bas', 6: 'bas', 
        8: 'gauche', 9: 'haut', 10: 'droite', 11: 'haut'
    }
    policy = {s: initial_policy_map.get(s, 'haut') for s in valid_states}
    
    with open(output_filename, 'w', encoding='utf-8') as log:
        log.write("--Initiation de la politique---\n\n")
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                if s in valid_states:
                    log.write(f"Grid_{r}_{c} -> {FULL_ACTION[policy[s]]} a été choisie initialement\n")
        log.write("\n")
        
        print_visualisation(grid, policy, rows, cols, log)
        
        iteration = 0
        while True:
            log.write(f"--- Itération {iteration} ---\n\n")
            log.write("---Evaluation de la politique (Résolution exacte par système d'équations)---\n\n")
            
            U = evaluate_policy(policy, grid, rows, cols, gamma, valid_states)
            
            def get_U(state):
                return U[state] if state in valid_states else 0.0

            for r in range(rows):
                for c in range(cols):
                    s = r * cols + c
                    if s in valid_states:
                        log.write(f"Grid_{r}_{c} (-> {FULL_ACTION[policy[s]]}): {U[s]}\n")
            
            log.write("\n---Amélioration de la politique---\n\n")
            policy_changed = False
            new_policy = {}
            
            for r in range(rows):
                for c in range(cols):
                    s = r * cols + c
                    if s in valid_states:
                        log.write(f"    Grid_{r}_{c}:\n")
                        q_values = {}
                        
                        for action in ACTIONS:
                            intended_next = apply_move(s, action, grid, rows, cols)
                            orth_actions = get_orthogonal_actions(action)
                            orth1_next = apply_move(s, orth_actions[0], grid, rows, cols)
                            orth2_next = apply_move(s, orth_actions[1], grid, rows, cols)
                            
                            transitions = {}
                            transitions[intended_next] = transitions.get(intended_next, 0.0) + 0.8
                            transitions[orth1_next] = transitions.get(orth1_next, 0.0) + 0.1
                            transitions[orth2_next] = transitions.get(orth2_next, 0.0) + 0.1
                            
                            q_total = 0.0
                            for next_s, prob in transitions.items():
                                r_val = get_reward(next_s, grid, cols)
                                q_total += prob * (r_val + gamma * get_U(next_s))
                                
                            q_values[action] = q_total
                        
                        current_action = policy[s]
                        log.write(f"Actuelle (-> {FULL_ACTION[current_action]}) : {q_values[current_action]}\n")
                        
                        for action in ACTIONS:
                            if action != current_action:
                                log.write(f"-> {FULL_ACTION[action]} : {q_values[action]}\n")
                                
                        best_action = max(q_values, key=q_values.get)
                        
                        if q_values[best_action] > q_values[current_action] + 1e-8:
                            new_policy[s] = best_action
                            policy_changed = True
                            log.write(f"\n Changement de politique : Grid_{r}_{c} -> {FULL_ACTION[best_action]}\n\n")
                        else:
                            new_policy[s] = current_action
                            log.write(f"\n Politique : Grid_{r}_{c} -> {FULL_ACTION[current_action]}\n\n")

            policy = new_policy
            
            print_visualisation(grid, policy, rows, cols, log)
            
            if not policy_changed:
                log.write("Pas de Changement : Fin de l'algorithme\n")
                log.write(f"Nombre d'itérations finales : {iteration}\n")
                break
                
            iteration += 1

if __name__ == "__main__":
    solve_policy_iteration()