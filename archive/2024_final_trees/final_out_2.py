# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 125 with ccp_alpha: 0.001
# Training score:  0.8638576666666666

# Testing score:  0.847

def predict_2(rho, theta, psi, vOwn, vInt):
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
