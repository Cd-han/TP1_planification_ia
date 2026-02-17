import random

ACTIONS = ['haut', 'bas', 'gauche', 'droite']

def get_orthogonal_actions(action):
    """Retourne les directions perpendiculaires pour la dérive stochastique."""
    if action in ['haut', 'bas']:
        return ['gauche', 'droite']
    else:
        return ['haut', 'bas']

def simulate_environment(state, action, grid, rows, cols):
    """
    Simule l'environnement avec sa part d'incertitude.
    Probabilités : 80% direction voulue, 10% dérive orthogonale 1, 10% dérive orthogonale 2.
    """
    rand = random.random()
    if rand < 0.8:
        actual_action = action
    elif rand < 0.9:
        actual_action = get_orthogonal_actions(action)[0]
    else:
        actual_action = get_orthogonal_actions(action)[1]
        
    r = state // cols
    c = state % cols
    
    nr, nc = r, c
    if actual_action == 'haut': nr -= 1
    elif actual_action == 'bas': nr += 1
    elif actual_action == 'gauche': nc -= 1
    elif actual_action == 'droite': nc += 1
        
    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state
    
    next_state = nr * cols + nc
    if grid[nr][nc] == 3:
        return state
        
    return next_state

def get_reward(state, grid, cols):
    """Renvoie la récompense associée à un état CIBLE."""
    r = state // cols
    c = state % cols
    cell = grid[r][c]
    
    if cell == 1:   return 1.0
    elif cell == 2: return -1.0
    else:           return -0.04

def is_terminal(state, grid, cols):
    """Vérifie si la case est le but (1) ou le fantôme (2)."""
    r = state // cols
    c = state % cols
    return grid[r][c] in [1, 2]

def choose_action(state, Q):
    """Sélectionne l'action gloutonne. S'il y a égalité parfaite, tranche aléatoirement."""
    q_values = [Q[state][a] for a in ACTIONS]
    max_q = max(q_values)
    best_actions = [a for a in ACTIONS if Q[state][a] == max_q]
    return random.choice(best_actions), q_values

def solve_q_learning(input_filename="Q-Learning.txt", output_filename="log-file_QL.txt"):
    try:
        with open(input_filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Erreur : Le fichier {input_filename} est introuvable.")
        return

    grid = []
    gamma = None
    alpha = None
    num_episodes = None
    
    for line in lines:
        if ',' in line:
            grid.append([int(x) for x in line.split(',')])
        elif gamma is None:
            gamma = float(line)
        elif alpha is None:
            alpha = float(line)
        else:
            num_episodes = int(line)
            
    rows = len(grid)
    cols = len(grid[0])
    num_states = rows * cols
    
    Q = {s: {a: 0.0 for a in ACTIONS} for s in range(num_states)}
    
    start_state = (rows - 1) * cols 
    
    with open(output_filename, 'w', encoding='utf-8') as log:
        
        for episode in range(1, num_episodes + 1):
            log.write(f"Itération {episode}\n")
            log.write(f"Départ de l'état S{start_state}\n\n")
            
            s = start_state
            step_count = 0
            max_steps_per_episode = 200 
            
            while not is_terminal(s, grid, cols) and step_count < max_steps_per_episode:
                step_count += 1
                
                action, q_vals = choose_action(s, Q)
                
                log.write(f"Action à prendre pi(S{s}) = argmax{{ Q(S{s}, haut), Q(S{s}, bas), Q(S{s}, gauche), Q(S{s}, droite)}}\n")
                log.write(f"\t\t\t= argmax{{ {q_vals[0]}, {q_vals[1]}, {q_vals[2]}, {q_vals[3]} }}\n")
                log.write(f"\t\t\t= {action}\n")
                
                next_state = simulate_environment(s, action, grid, rows, cols)
                log.write(f"S{s} -> S{next_state}\n\n")
                
                R = get_reward(next_state, grid, cols)
                q_old = Q[s][action]
                
                if is_terminal(next_state, grid, cols):
                    max_q_next = 0.0
                    Q[s][action] = q_old + alpha * (R + gamma * max_q_next - q_old)
                    
                    log.write(f"Q(S{s},{action}) <- Q(S{s},{action}) + α * (R(S{next_state}) + γ * max{{ Q(S{next_state}, None) }} - Q(S{s},{action}))\n")
                    log.write(f"\t\t\t = {q_old} + {alpha} * ({R} + {gamma} * {max_q_next} - {q_old})\n")
                    log.write(f"\t\t\t = {Q[s][action]}\n\n")
                    
                    log.write("Fin de l'essai\n\n\n")
                    break
                else:
                    q_next_vals = [Q[next_state][a] for a in ACTIONS]
                    max_q_next = max(q_next_vals)
                    Q[s][action] = q_old + alpha * (R + gamma * max_q_next - q_old)
                    
                    log.write(f"Q(S{s},{action}) <- Q(S{s},{action}) + α * (R(S{next_state}) + γ * max{{ Q(S{next_state}, haut), Q(S{next_state}, bas), Q(S{next_state}, gauche), Q(S{next_state}, droite) }} - Q(S{s},{action}))\n")
                    log.write(f"\t\t\t = {q_old} + {alpha} * ({R} + {gamma} * {max_q_next} - {q_old})\n")
                    log.write(f"\t\t\t = {Q[s][action]}\n\n")
                    
                    s = next_state
                    
            if step_count >= max_steps_per_episode:
                log.write("Arrêt prématuré de l'essai (limite de déplacements atteinte).\n\n\n")

        log.write("/**************************/\n")
        log.write("Meilleure action pour chaque état :\n")
        
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                s = r * cols + c
                if grid[r][c] == 3: 
                    row_str += f"{'None':<11}"
                elif grid[r][c] in [1, 2]: 
                    row_str += f"{'None':<11}"
                else:
                    best_action, _ = choose_action(s, Q)
                    row_str += f"{best_action:<11}"
            log.write(row_str + "\n")

if __name__ == "__main__":
    solve_q_learning()