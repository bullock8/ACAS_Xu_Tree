# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 123 with ccp_alpha: 0.001
# Training score:  0.8749323333333333

# Testing score:  0.827

def predict_1(rho, theta, psi, vOwn, vInt):
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
