if ego.timer >= acas_update_time:
    if ego.agent_mode == CraftMode.Coc:  # advisory 0
        if rho <= 15545.62:
            if theta <= -0.13:
                if rho <= 1727.29:
                    if psi <= 0.13:
                        if rho <= 1320.87:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if rho > 1320.87:
                            if theta <= -1.78:
                                if psi <= -0.95:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if psi > -0.95:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if theta > -1.78:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if psi > 0.13:
                        if rho <= 711.24:
                            if psi <= 2.22:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if psi > 2.22:
                                if theta <= -1.97:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if theta > -1.97:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                        if rho > 711.24:
                            if theta <= -2.73:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if theta > -2.73:
                                if theta <= -0.63:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if theta > -0.63:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                if rho > 1727.29:
                    if theta <= -0.83:
                        if psi <= -0.25:
                            if rho <= 5385.08:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                            if rho > 5385.08:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                        if psi > -0.25:
                            if theta <= -2.73:
                                if psi <= 0.38:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if psi > 0.38:
                                    if rho <= 5385.08:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                    if rho > 5385.08:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                            if theta > -2.73:
                                if rho <= 6197.93:
                                    if psi <= 1.9:
                                        if theta <= -2.03:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                                        if theta > -2.03:
                                            if rho <= 4165.82:
                                                next.agent_mode = CraftMode.Strong_left
                                                next.timer = 0
                                            if rho > 4165.82:
                                                if psi <= 0.7:
                                                    next.agent_mode = CraftMode.Weak_left
                                                    next.timer = 0
                                                if psi > 0.7:
                                                    next.agent_mode = CraftMode.Strong_left
                                                    next.timer = 0
                                    if psi > 1.9:
                                        if theta <= -1.14:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                                        if theta > -1.14:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                if rho > 6197.93:
                                    if theta <= -1.97:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if theta > -1.97:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                    if theta > -0.83:
                        if rho <= 4572.24:
                            if psi <= 0.76:
                                if theta <= -0.57:
                                    if rho <= 3352.98:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if rho > 3352.98:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                if theta > -0.57:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if psi > 0.76:
                                if theta <= -0.44:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if theta > -0.44:
                                    if psi <= 2.54:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if psi > 2.54:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                        if rho > 4572.24:
                            if theta <= -0.38:
                                if rho <= 9449.3:
                                    if psi <= 1.08:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if psi > 1.08:
                                        if rho <= 7417.19:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                        if rho > 7417.19:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                                if rho > 9449.3:
                                    if psi <= 1.21:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if psi > 1.21:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                            if theta > -0.38:
                                if psi <= 0.83:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if psi > 0.83:
                                    if psi <= 2.6:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if psi > 2.6:
                                        if rho <= 10262.14:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                        if rho > 10262.14:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
            if theta > -0.13:
                if rho <= 1727.29:
                    if psi <= -0.19:
                        if rho <= 508.03:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if rho > 508.03:
                            if theta <= 0.44:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if theta > 0.44:
                                if theta <= 2.73:
                                    if psi <= -1.78:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if psi > -1.78:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                if theta > 2.73:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                    if psi > -0.19:
                        if rho <= 1524.08:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                        if rho > 1524.08:
                            if theta <= 1.71:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if theta > 1.71:
                                if psi <= 1.14:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > 1.14:
                                    next.agent_mode = CraftMode.Weak_right
                                    next.timer = 0
                if rho > 1727.29:
                    if theta <= 1.33:
                        if rho <= 5181.87:
                            if psi <= -0.32:
                                if theta <= 0.44:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if theta > 0.44:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                            if psi > -0.32:
                                if theta <= 0.57:
                                    if rho <= 3759.4:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if rho > 3759.4:
                                        if psi <= 1.59:
                                            next.agent_mode = CraftMode.Weak_right
                                            next.timer = 0
                                        if psi > 1.59:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                if theta > 0.57:
                                    if rho <= 3556.19:
                                        if rho <= 2133.71:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if rho > 2133.71:
                                            next.agent_mode = CraftMode.Weak_right
                                            next.timer = 0
                                    if rho > 3556.19:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                        if rho > 5181.87:
                            if psi <= -1.08:
                                if theta <= 0.38:
                                    if rho <= 9449.3:
                                        if psi <= -2.22:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                        if psi > -2.22:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                                    if rho > 9449.3:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                if theta > 0.38:
                                    if rho <= 8026.82:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if rho > 8026.82:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                            if psi > -1.08:
                                if rho <= 9652.51:
                                    if theta <= 0.13:
                                        if psi <= 2.22:
                                            if psi <= 0.13:
                                                next.agent_mode = CraftMode.Weak_left
                                                next.timer = 0
                                            if psi > 0.13:
                                                next.agent_mode = CraftMode.Weak_right
                                                next.timer = 0
                                        if psi > 2.22:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                    if theta > 0.13:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                if rho > 9652.51:
                                    if psi <= 2.09:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if psi > 2.09:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                    if theta > 1.33:
                        if rho <= 8636.46:
                            if psi <= 0.38:
                                if theta <= 2.73:
                                    if theta <= 1.9:
                                        if rho <= 2336.92:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if rho > 2336.92:
                                            if rho <= 5791.51:
                                                if psi <= -1.59:
                                                    next.agent_mode = CraftMode.Weak_right
                                                    next.timer = 0
                                                if psi > -1.59:
                                                    next.agent_mode = CraftMode.Strong_right
                                                    next.timer = 0
                                            if rho > 5791.51:
                                                next.agent_mode = CraftMode.Weak_right
                                                next.timer = 0
                                    if theta > 1.9:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                if theta > 2.73:
                                    if rho <= 3352.98:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if rho > 3352.98:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                            if psi > 0.38:
                                if rho <= 4369.03:
                                    next.agent_mode = CraftMode.Weak_right
                                    next.timer = 0
                                if rho > 4369.03:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                        if rho > 8636.46:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
        if rho > 15545.62:
            if rho <= 18593.78:
                if psi <= 2.03:
                    next.agent_mode = CraftMode.Coc
                    next.timer = 0
                if psi > 2.03:
                    if theta <= -0.19:
                        next.agent_mode = CraftMode.Weak_left
                        next.timer = 0
                    if theta > -0.19:
                        if theta <= 0.25:
                            next.agent_mode = CraftMode.Weak_right
                            next.timer = 0
                        if theta > 0.25:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
            if rho > 18593.78:
                next.agent_mode = CraftMode.Coc
                next.timer = 0
    
    if ego.agent_mode == CraftMode.Weak_Left: # advisory 1
        if psi <= -2.79:
            if theta <= 0.32:
                if rho <= 13513.51:
                    if theta <= 0.06:
                        if rho <= 1930.5:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if rho > 1930.5:
                            if theta <= -0.95:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                            if theta > -0.95:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if theta > 0.06:
                        next.agent_mode = CraftMode.Strong_right
                        next.timer = 0
                if rho > 13513.51:
                    if theta <= -0.32:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0
                    if theta > -0.32:
                        next.agent_mode = CraftMode.Weak_left
                        next.timer = 0
            if theta > 0.32:
                if rho <= 10058.93:
                    if theta <= 1.27:
                        next.agent_mode = CraftMode.Strong_right
                        next.timer = 0
                    if theta > 1.27:
                        if rho <= 1727.29:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if rho > 1727.29:
                            next.agent_mode = CraftMode.Weak_left
                            next.timer = 0
                if rho > 10058.93:
                    if rho <= 10871.77:
                        next.agent_mode = CraftMode.Weak_right
                        next.timer = 0
                    if rho > 10871.77:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0
        if psi > -2.79:
            if rho <= 9042.88:
                if theta <= -0.51:
                    if rho <= 1930.5:
                        if psi <= 0.13:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if psi > 0.13:
                            if psi <= 1.46:
                                if theta <= -0.89:
                                    if theta <= -1.52:
                                        if rho <= 1320.87:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if rho > 1320.87:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                    if theta > -1.52:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                if theta > -0.89:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                            if psi > 1.46:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if rho > 1930.5:
                        if psi <= -0.57:
                            if rho <= 5588.29:
                                if theta <= -0.83:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if theta > -0.83:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                            if rho > 5588.29:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                        if psi > -0.57:
                            if theta <= -1.33:
                                if rho <= 3556.19:
                                    if psi <= 1.78:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if psi > 1.78:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                if rho > 3556.19:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                            if theta > -1.33:
                                if rho <= 6807.56:
                                    if theta <= -0.76:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if theta > -0.76:
                                        if psi <= 2.03:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if psi > 2.03:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                if rho > 6807.56:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                if theta > -0.51:
                    if psi <= -0.25:
                        if theta <= 0.38:
                            if rho <= 4165.82:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if rho > 4165.82:
                                if psi <= -1.33:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if psi > -1.33:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                        if theta > 0.38:
                            if rho <= 1524.08:
                                if rho <= 914.45:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if rho > 914.45:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if rho > 1524.08:
                                if theta <= 2.03:
                                    if theta <= 0.51:
                                        if psi <= -2.03:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if psi > -2.03:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                    if theta > 0.51:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                if theta > 2.03:
                                    if rho <= 3962.61:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if rho > 3962.61:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                    if psi > -0.25:
                        if theta <= 1.21:
                            if theta <= -0.19:
                                if psi <= 2.54:
                                    if rho <= 6604.35:
                                        if psi <= 0.25:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                        if psi > 0.25:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                    if rho > 6604.35:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                if psi > 2.54:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if theta > -0.19:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                        if theta > 1.21:
                            if rho <= 1930.5:
                                if psi <= 0.44:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > 0.44:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if rho > 1930.5:
                                if rho <= 3962.61:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if rho > 3962.61:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
            if rho > 9042.88:
                if theta <= 0.76:
                    if theta <= -1.08:
                        if rho <= 13513.51:
                            if theta <= -1.78:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if theta > -1.78:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                        if rho > 13513.51:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                    if theta > -1.08:
                        if psi <= 0.57:
                            if psi <= -1.33:
                                if theta <= -0.19:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                                if theta > -0.19:
                                    if rho <= 13107.09:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if rho > 13107.09:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                            if psi > -1.33:
                                if psi <= 0.32:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                                if psi > 0.32:
                                    if theta <= -0.51:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if theta > -0.51:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                        if psi > 0.57:
                            if theta <= 0.06:
                                if rho <= 11684.62:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if rho > 11684.62:
                                    if theta <= -0.19:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if theta > -0.19:
                                        if psi <= 2.35:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                        if psi > 2.35:
                                            if rho <= 16358.46:
                                                next.agent_mode = CraftMode.Strong_left
                                                next.timer = 0
                                            if rho > 16358.46:
                                                next.agent_mode = CraftMode.Weak_left
                                                next.timer = 0
                            if theta > 0.06:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                if theta > 0.76:
                    next.agent_mode = CraftMode.Coc
                    next.timer = 0
    
    if ego.agent_mode == CraftMode.Weak_Right: # advisory 2
        if rho <= 7213.98:
            if theta <= 0.44:
                if theta <= -2.22:
                    if rho <= 1524.08:
                        if psi <= 0.32:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if psi > 0.32:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                    if rho > 1524.08:
                        if rho <= 4572.24:
                            next.agent_mode = CraftMode.Weak_right
                            next.timer = 0
                        if rho > 4572.24:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                if theta > -2.22:
                    if theta <= -0.63:
                        if rho <= 1117.66:
                            if psi <= 1.9:
                                if psi <= -1.33:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > -1.33:
                                    if theta <= -1.52:
                                        if psi <= 0.57:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                        if psi > 0.57:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                    if theta > -1.52:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                            if psi > 1.9:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                        if rho > 1117.66:
                            if psi <= -0.13:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                            if psi > -0.13:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if theta > -0.63:
                        if psi <= -0.19:
                            if theta <= 0.25:
                                if rho <= 3149.77:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if rho > 3149.77:
                                    if psi <= -1.33:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if psi > -1.33:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                            if theta > 0.25:
                                if psi <= -2.6:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > -2.6:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                        if psi > -0.19:
                            if theta <= -0.32:
                                if psi <= 2.03:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > 2.03:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if theta > -0.32:
                                if rho <= 3759.4:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if rho > 3759.4:
                                    if psi <= 1.27:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                    if psi > 1.27:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
            if theta > 0.44:
                if rho <= 1727.29:
                    if psi <= -0.19:
                        if psi <= -1.84:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                        if psi > -1.84:
                            if theta <= 0.83:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if theta > 0.83:
                                if theta <= 2.48:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if theta > 2.48:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                    if psi > -0.19:
                        next.agent_mode = CraftMode.Strong_right
                        next.timer = 0
                if rho > 1727.29:
                    if theta <= 1.52:
                        if psi <= 0.0:
                            if theta <= 0.7:
                                if psi <= -2.22:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > -2.22:
                                    if psi <= -1.08:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if psi > -1.08:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                            if theta > 0.7:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                        if psi > 0.0:
                            if rho <= 2743.34:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if rho > 2743.34:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                    if theta > 1.52:
                        if psi <= 0.25:
                            if psi <= -1.33:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                            if psi > -1.33:
                                if rho <= 4572.24:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if rho > 4572.24:
                                    next.agent_mode = CraftMode.Weak_right
                                    next.timer = 0
                        if psi > 0.25:
                            if rho <= 4978.66:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                            if rho > 4978.66:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
        if rho > 7213.98:
            if rho <= 8839.67:
                if theta <= -0.51:
                    if theta <= -0.83:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0
                    if theta > -0.83:
                        if psi <= 2.48:
                            if psi <= 1.71:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if psi > 1.71:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                        if psi > 2.48:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                if theta > -0.51:
                    if theta <= 1.02:
                        if psi <= -2.16:
                            if theta <= 0.32:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if theta > 0.32:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                        if psi > -2.16:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                    if theta > 1.02:
                        if psi <= 0.13:
                            next.agent_mode = CraftMode.Weak_right
                            next.timer = 0
                        if psi > 0.13:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
            if rho > 8839.67:
                if theta <= -1.02:
                    next.agent_mode = CraftMode.Coc
                    next.timer = 0
                if theta > -1.02:
                    if theta <= 0.95:
                        if rho <= 12294.25:
                            next.agent_mode = CraftMode.Weak_right
                            next.timer = 0
                        if rho > 12294.25:
                            if psi <= 0.89:
                                if psi <= -1.59:
                                    if theta <= -0.13:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if theta > -0.13:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                if psi > -1.59:
                                    if rho <= 29363.95:
                                        if psi <= 0.25:
                                            if theta <= 0.57:
                                                next.agent_mode = CraftMode.Coc
                                                next.timer = 0
                                            if theta > 0.57:
                                                next.agent_mode = CraftMode.Weak_right
                                                next.timer = 0
                                        if psi > 0.25:
                                            if theta <= -0.38:
                                                next.agent_mode = CraftMode.Weak_right
                                                next.timer = 0
                                            if theta > -0.38:
                                                next.agent_mode = CraftMode.Coc
                                                next.timer = 0
                                    if rho > 29363.95:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                            if psi > 0.89:
                                if theta <= 0.19:
                                    if theta <= -0.76:
                                        if psi <= 1.59:
                                            next.agent_mode = CraftMode.Weak_right
                                            next.timer = 0
                                        if psi > 1.59:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                    if theta > -0.76:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                if theta > 0.19:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                    if theta > 0.95:
                        if rho <= 13919.93:
                            if psi <= -0.19:
                                if theta <= 1.9:
                                    next.agent_mode = CraftMode.Weak_right
                                    next.timer = 0
                                if theta > 1.9:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                            if psi > -0.19:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                        if rho > 13919.93:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
    
    if ego.agent_mode == CraftMode.Strong_Left: # advisory 3
        if rho <= 8839.67:
            if rho <= 4775.45:
                if theta <= -0.44:
                    if psi <= 0.0:
                        next.agent_mode = CraftMode.Strong_left
                        next.timer = 0
                    if psi > 0.0:
                        if rho <= 914.45:
                            if psi <= 1.4:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if psi > 1.4:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                        if rho > 914.45:
                            if theta <= -0.63:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if theta > -0.63:
                                if psi <= 1.84:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > 1.84:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                if theta > -0.44:
                    if rho <= 711.24:
                        if psi <= -0.19:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if psi > -0.19:
                            if theta <= 1.02:
                                if psi <= 2.16:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > 2.16:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if theta > 1.02:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if rho > 711.24:
                        if theta <= 0.13:
                            if psi <= -0.06:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if psi > -0.06:
                                if psi <= 2.67:
                                    if psi <= 0.63:
                                        if rho <= 2540.13:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if rho > 2540.13:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                    if psi > 0.63:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                if psi > 2.67:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                        if theta > 0.13:
                            if theta <= 2.35:
                                if theta <= 0.51:
                                    if psi <= -0.44:
                                        if psi <= -2.41:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if psi > -2.41:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                    if psi > -0.44:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                if theta > 0.51:
                                    if psi <= 0.38:
                                        if rho <= 1320.87:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if rho > 1320.87:
                                            if psi <= -1.08:
                                                if theta <= 1.65:
                                                    next.agent_mode = CraftMode.Strong_right
                                                    next.timer = 0
                                                if theta > 1.65:
                                                    next.agent_mode = CraftMode.Weak_left
                                                    next.timer = 0
                                            if psi > -1.08:
                                                next.agent_mode = CraftMode.Strong_right
                                                next.timer = 0
                                    if psi > 0.38:
                                        if theta <= 1.14:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if theta > 1.14:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                            if theta > 2.35:
                                if rho <= 1524.08:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if rho > 1524.08:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
            if rho > 4775.45:
                if theta <= 0.95:
                    if theta <= -0.32:
                        if theta <= -1.9:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                        if theta > -1.9:
                            if psi <= 0.83:
                                if psi <= -0.38:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if psi > -0.38:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                            if psi > 0.83:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if theta > -0.32:
                        if psi <= 1.65:
                            if theta <= 0.38:
                                if psi <= -2.67:
                                    if theta <= 0.06:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if theta > 0.06:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                if psi > -2.67:
                                    if psi <= -1.21:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if psi > -1.21:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                            if theta > 0.38:
                                if psi <= -1.08:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > -1.08:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                        if psi > 1.65:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                if theta > 0.95:
                    if psi <= -0.89:
                        if theta <= 1.78:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                        if theta > 1.78:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                    if psi > -0.89:
                        if theta <= 1.4:
                            if psi <= -0.25:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                            if psi > -0.25:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                        if theta > 1.4:
                            if psi <= -0.38:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if psi > -0.38:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
        if rho > 8839.67:
            if theta <= -1.4:
                next.agent_mode = CraftMode.Coc
                next.timer = 0
            if theta > -1.4:
                if theta <= 1.14:
                    if rho <= 13919.93:
                        if psi <= -1.59:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if psi > -1.59:
                            if psi <= 2.09:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                            if psi > 2.09:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                    if rho > 13919.93:
                        if rho <= 35257.06:
                            if psi <= 0.51:
                                if theta <= -0.19:
                                    if psi <= -1.27:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if psi > -1.27:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                if theta > -0.19:
                                    if psi <= -0.32:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if psi > -0.32:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                            if psi > 0.51:
                                if theta <= 0.25:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                                if theta > 0.25:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                        if rho > 35257.06:
                            if theta <= -1.02:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if theta > -1.02:
                                if psi <= 1.21:
                                    if psi <= -0.7:
                                        if theta <= -0.19:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                        if theta > -0.19:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                                    if psi > -0.7:
                                        if theta <= -0.76:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                                        if theta > -0.76:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                if psi > 1.21:
                                    if theta <= 0.19:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if theta > 0.19:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                if theta > 1.14:
                    if theta <= 1.9:
                        if rho <= 39727.69:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                        if rho > 39727.69:
                            if psi <= 0.06:
                                if psi <= -0.95:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                                if psi > -0.95:
                                    next.agent_mode = CraftMode.Weak_left
                                    next.timer = 0
                            if psi > 0.06:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                    if theta > 1.9:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0

    
    if ego.agent_mode == CraftMode.Strong_Right: # advisory 4
        if rho <= 11074.98:
            if theta <= -1.08:
                if rho <= 4978.66:
                    if rho <= 1524.08:
                        if psi <= 0.51:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if psi > 0.51:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                    if rho > 1524.08:
                        if psi <= -0.19:
                            if theta <= -1.59:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if theta > -1.59:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                        if psi > -0.19:
                            if psi <= 1.52:
                                if theta <= -2.35:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if theta > -2.35:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if psi > 1.52:
                                if theta <= -1.46:
                                    next.agent_mode = CraftMode.Weak_right
                                    next.timer = 0
                                if theta > -1.46:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                if rho > 4978.66:
                    if theta <= -1.84:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0
                    if theta > -1.84:
                        if psi <= 1.14:
                            if psi <= 0.19:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if psi > 0.19:
                                if rho <= 5791.51:
                                    if psi <= 0.32:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if psi > 0.32:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                if rho > 5791.51:
                                    if psi <= 0.76:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if psi > 0.76:
                                        if rho <= 7417.19:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                        if rho > 7417.19:
                                            next.agent_mode = CraftMode.Weak_left
                                            next.timer = 0
                        if psi > 1.14:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
            if theta > -1.08:
                if theta <= 0.51:
                    if psi <= -0.06:
                        if theta <= -0.57:
                            if rho <= 4978.66:
                                if rho <= 1727.29:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if rho > 1727.29:
                                    if psi <= -2.41:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if psi > -2.41:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                            if rho > 4978.66:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                        if theta > -0.57:
                            if theta <= 0.25:
                                next.agent_mode = CraftMode.Strong_left
                                next.timer = 0
                            if theta > 0.25:
                                if rho <= 7213.98:
                                    if psi <= -2.73:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if psi > -2.73:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                if rho > 7213.98:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                    if psi > -0.06:
                        if theta <= -0.25:
                            if rho <= 1524.08:
                                if psi <= 0.32:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if psi > 0.32:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                            if rho > 1524.08:
                                if theta <= -1.02:
                                    if psi <= 0.57:
                                        next.agent_mode = CraftMode.Weak_left
                                        next.timer = 0
                                    if psi > 0.57:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                if theta > -1.02:
                                    if theta <= -0.57:
                                        next.agent_mode = CraftMode.Strong_left
                                        next.timer = 0
                                    if theta > -0.57:
                                        if psi <= 2.28:
                                            if psi <= 0.57:
                                                next.agent_mode = CraftMode.Strong_left
                                                next.timer = 0
                                            if psi > 0.57:
                                                next.agent_mode = CraftMode.Strong_right
                                                next.timer = 0
                                        if psi > 2.28:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                        if theta > -0.25:
                            if theta <= -0.13:
                                if psi <= 2.79:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if psi > 2.79:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                            if theta > -0.13:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                if theta > 0.51:
                    if rho <= 2743.34:
                        if psi <= -0.32:
                            if psi <= -1.71:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if psi > -1.71:
                                if theta <= 0.89:
                                    next.agent_mode = CraftMode.Strong_left
                                    next.timer = 0
                                if theta > 0.89:
                                    if rho <= 1117.66:
                                        if psi <= -1.08:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                                        if psi > -1.08:
                                            next.agent_mode = CraftMode.Strong_left
                                            next.timer = 0
                                    if rho > 1117.66:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                        if psi > -0.32:
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                    if rho > 2743.34:
                        if psi <= 0.57:
                            if rho <= 7417.19:
                                if theta <= 1.4:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                                if theta > 1.4:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                            if rho > 7417.19:
                                if psi <= -1.59:
                                    if theta <= 1.21:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if theta > 1.21:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                if psi > -1.59:
                                    next.agent_mode = CraftMode.Weak_right
                                    next.timer = 0
                        if psi > 0.57:
                            if rho <= 4775.45:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                            if rho > 4775.45:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
        if rho > 11074.98:
            if theta <= -1.27:
                next.agent_mode = CraftMode.Coc
                next.timer = 0
            if theta > -1.27:
                if theta <= 1.33:
                    if theta <= 0.51:
                        if psi <= 0.57:
                            if theta <= -0.19:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if theta > -0.19:
                                if psi <= -1.21:
                                    if rho <= 15748.83:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if rho > 15748.83:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                if psi > -1.21:
                                    if rho <= 35053.85:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if rho > 35053.85:
                                        if psi <= -0.38:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                        if psi > -0.38:
                                            next.agent_mode = CraftMode.Strong_right
                                            next.timer = 0
                        if psi > 0.57:
                            if theta <= 0.13:
                                if theta <= 0.0:
                                    if theta <= -0.7:
                                        if psi <= 1.9:
                                            next.agent_mode = CraftMode.Weak_right
                                            next.timer = 0
                                        if psi > 1.9:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                    if theta > -0.7:
                                        if psi <= 1.4:
                                            next.agent_mode = CraftMode.Coc
                                            next.timer = 0
                                        if psi > 1.4:
                                            next.agent_mode = CraftMode.Weak_right
                                            next.timer = 0
                                if theta > 0.0:
                                    if psi <= 2.54:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                    if psi > 2.54:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                            if theta > 0.13:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                    if theta > 0.51:
                        if psi <= -0.25:
                            if psi <= -2.22:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if psi > -2.22:
                                if psi <= -0.89:
                                    if theta <= 1.02:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                                    if theta > 1.02:
                                        next.agent_mode = CraftMode.Coc
                                        next.timer = 0
                                if psi > -0.89:
                                    if theta <= 0.95:
                                        next.agent_mode = CraftMode.Strong_right
                                        next.timer = 0
                                    if theta > 0.95:
                                        next.agent_mode = CraftMode.Weak_right
                                        next.timer = 0
                        if psi > -0.25:
                            if psi <= 0.76:
                                if rho <= 20625.89:
                                    next.agent_mode = CraftMode.Coc
                                    next.timer = 0
                                if rho > 20625.89:
                                    next.agent_mode = CraftMode.Strong_right
                                    next.timer = 0
                            if psi > 0.76:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                if theta > 1.33:
                    if rho <= 19203.41:
                        if psi <= 0.25:
                            if psi <= -0.7:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                            if psi > -0.7:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                        if psi > 0.25:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                    if rho > 19203.41:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0