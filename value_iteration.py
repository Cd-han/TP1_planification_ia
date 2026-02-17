import math

ACTIONS = ['haut', 'bas', 'gauche', 'droite']

def get_orthogonal_actions(action):
    if action in ['haut', 'bas']:
        return ['gauche', 'droite']
    else:
        return ['haut', 'bas']

def apply_move(state, action, grid, rows, cols):
    """Calcule l'état d'arrivée après un mouvement (incluant les rebonds)."""
    r = state // cols
    c = state % cols
    
    nr, nc = r, c
    if action == 'haut':
        nr -= 1
    elif action == 'bas':
        nr += 1
    elif action == 'gauche':
        nc -= 1
    elif action == 'droite':
        nc += 1
        
    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state
    
    next_state = nr * cols + nc
    if grid[nr][nc] == 3:
        return state
        
    return next_state

def get_reward(state, grid, cols):
    """Renvoie la récompense associée à un état cible."""
    r = state // cols
    c = state % cols
    cell = grid[r][c]
    
    if cell == 1:   
        return 1.0
    elif cell == 2: 
        return -1.0
    else:          
        return -0.04

def format_q_calc(r, gamma, u):
    """Formate la chaîne de calcul pour qu'elle corresponde exactement à la trace."""
    s = f"[{r:.2f}+{gamma}*{u:.3f}]"
    return s.replace("-0.04+", "-0.04+").replace("1.00+", "1.0+").replace("-1.00+", "-1.0+")

def solve_value_iteration(input_filename="value-iteration.txt", output_filename="log-file_VI.txt"):
    try:
        with open(input_filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Erreur : Le fichier {input_filename} est introuvable.")
        return

    grid = []
    gamma = 0.0
    tolerance = 0.0
    
    for line in lines:
        if ',' in line:
            grid.append([int(x) for x in line.split(',')])
        elif gamma == 0.0:
            gamma = float(line)
        else:
            tolerance = float(line)
            
    rows = len(grid)
    cols = len(grid[0])
    num_states = rows * cols
    
    U = [0.0] * num_states
    
    walls = [i for i in range(num_states) if grid[i // cols][i % cols] == 3]
    terminals = [i for i in range(num_states) if grid[i // cols][i % cols] in [1, 2]]
    
    with open(output_filename, 'w', encoding='utf-8') as log:
        
        iteration = 1
        while True:
            log.write(f"Itération {iteration} :\n")
            U_prime = list(U)
            max_diff = 0.0
            sum_diff = 0.0

            for s in range(num_states):
                if s in walls or s in terminals:
                    continue
                    
                log.write(f"U'{s}: \n")
                q_values = {}
                
                for action in ACTIONS:
                    intended_next = apply_move(s, action, grid, rows, cols)
                    orth_actions = get_orthogonal_actions(action)
                    orth1_next = apply_move(s, orth_actions[0], grid, rows, cols)
                    orth2_next = apply_move(s, orth_actions[1], grid, rows, cols)
                    
                    r_intended = get_reward(intended_next, grid, cols)
                    r_orth1 = get_reward(orth1_next, grid, cols)
                    r_orth2 = get_reward(orth2_next, grid, cols)
                    
                    q_intended = 0.8 * (r_intended + gamma * U[intended_next])
                    q_orth1 = 0.1 * (r_orth1 + gamma * U[orth1_next])
                    q_orth2 = 0.1 * (r_orth2 + gamma * U[orth2_next])
                    
                    q_total = q_intended + q_orth1 + q_orth2
                    q_values[action] = q_total
                    
                    str_intended = f"0.8*{format_q_calc(r_intended, gamma, U[intended_next])}"
                    str_orth1 = f"0.1*{format_q_calc(r_orth1, gamma, U[orth1_next])}"
                    str_orth2 = f"0.1*{format_q_calc(r_orth2, gamma, U[orth2_next])}"
                    
                    log.write(f"Q(S{s},{action}) = {str_intended} + {str_orth1} + {str_orth2} = {q_total:.4f}\n")
                
                best_q = max(q_values.values())
                U_prime[s] = best_q
                
                q_list_str = ", ".join([f"{q_values[a]:.4f}" for a in ACTIONS])
                log.write(f"U'{s} = max{{{q_list_str}}} = {best_q:.4f}\n\n")
            
            log.write(f"UTILITES A L'ITERATION {iteration}:\n")
            for r in range(rows):
                row_str = ""
                for c in range(cols):
                    s = r * cols + c
                    if s in walls:
                        row_str += "  0.000 "
                    elif grid[r][c] == 1:
                        row_str += "  1.000 "
                    elif grid[r][c] == 2:
                        row_str += " -1.000 "
                    else:
                        row_str += f"{U_prime[s]:7.3f} "
                log.write(row_str + "\n")
            
            for s in range(num_states):
                if s not in walls and s not in terminals:
                    sum_diff += abs(U[s] - U_prime[s])
            
            log.write(f"\nSomme des differences |Us - U'(S)| = {sum_diff:.6f}\n\n")
            
            U = U_prime
            if sum_diff < tolerance:
                log.write(f"Difference < {tolerance} . Arret des itérations\n\n")
                break
            iteration += 1

        log.write("Recherche des actions optimales :\n\n")
        best_actions = {}
        for s in range(num_states):
            if s in walls or s in terminals:
                continue
            log.write(f"S{s}:\n")
            q_values = {}
            for action in ACTIONS:
                intended_next = apply_move(s, action, grid, rows, cols)
                orth_actions = get_orthogonal_actions(action)
                orth1_next = apply_move(s, orth_actions[0], grid, rows, cols)
                orth2_next = apply_move(s, orth_actions[1], grid, rows, cols)
                
                r_intended = get_reward(intended_next, grid, cols)
                r_orth1 = get_reward(orth1_next, grid, cols)
                r_orth2 = get_reward(orth2_next, grid, cols)
                
                q_intended = 0.8 * (r_intended + gamma * U[intended_next])
                q_orth1 = 0.1 * (r_orth1 + gamma * U[orth1_next])
                q_orth2 = 0.1 * (r_orth2 + gamma * U[orth2_next])
                
                q_total = q_intended + q_orth1 + q_orth2
                q_values[action] = q_total
                
                str_intended = f"0.8*{format_q_calc(r_intended, gamma, U[intended_next])}"
                str_orth1 = f"0.1*{format_q_calc(r_orth1, gamma, U[orth1_next])}"
                str_orth2 = f"0.1*{format_q_calc(r_orth2, gamma, U[orth2_next])}"
                
                log.write(f"Q(S{s},{action}) = {str_intended} + {str_orth1} + {str_orth2} = {q_total:.4f}\n")
                
            best_action = max(q_values, key=q_values.get)
            best_actions[s] = best_action
            q_list_str = ", ".join([f"{q_values[a]:.4f}" for a in ACTIONS])
            log.write(f"Meilleure action = argmax{{{q_list_str}}} = {best_action}\n\n")
            
        log.write("Meilleure action de chaque état :\n")
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                s = r * cols + c
                if grid[r][c] == 3:
                    row_str += "MUR      "
                elif grid[r][c] == 1:
                    row_str += "BUT      "
                elif grid[r][c] == 2:
                    row_str += "FANT     "
                else:
                    row_str += f"{best_actions[s]:<9}"
            log.write(row_str + "\n")
            
        log.write("\nPlan optimal:\n")
        current_state = (rows - 1) * cols 
        path = []
        visited = set()
        
        while current_state not in terminals and current_state not in visited:
            visited.add(current_state)
            action = best_actions[current_state]
            path.append(action)
            current_state = apply_move(current_state, action, grid, rows, cols)
            
        path.append("but")
        log.write(" -> ".join(path) + "\n")

if __name__ == "__main__":
    solve_value_iteration()