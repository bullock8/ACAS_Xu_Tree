# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 147 with ccp_alpha: 0.001
# Training score:  0.8545843333333333

# Testing score:  0.709

def predict_4(rho, theta, psi, vOwn, vInt):
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
