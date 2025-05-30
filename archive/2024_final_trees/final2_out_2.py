# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 61 with ccp_alpha: 0.002
# Training score:  0.8351386666666667

# Testing score:  0.867

def predict_2(rho, theta, psi, vOwn, vInt):
    if rho <= 7213.98:
        if theta <= 0.44:
            if theta <= -2.22:
                if rho <= 1524.08:
                    next.agent_mode = CraftMode.Strong_right
                    next.timer = 0
                if rho > 1524.08:
                    next.agent_mode = CraftMode.Weak_right
                    next.timer = 0
            if theta > -2.22:
                if theta <= -0.63:
                    if rho <= 1117.66:
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
                        next.agent_mode = CraftMode.Strong_right
                        next.timer = 0
                    if psi > 0.0:
                        next.agent_mode = CraftMode.Weak_right
                        next.timer = 0
                if theta > 1.52:
                    next.agent_mode = CraftMode.Weak_right
                    next.timer = 0
    if rho > 7213.98:
        if rho <= 8839.67:
            if theta <= -0.51:
                if theta <= -0.83:
                    next.agent_mode = CraftMode.Coc
                    next.timer = 0
                if theta > -0.83:
                    next.agent_mode = CraftMode.Weak_left
                    next.timer = 0
            if theta > -0.51:
                if theta <= 1.02:
                    next.agent_mode = CraftMode.Strong_right
                    next.timer = 0
                if theta > 1.02:
                    next.agent_mode = CraftMode.Weak_right
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
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                        if psi > 0.89:
                            if theta <= 0.19:
                                next.agent_mode = CraftMode.Weak_right
                                next.timer = 0
                            if theta > 0.19:
                                next.agent_mode = CraftMode.Coc
                                next.timer = 0
                if theta > 0.95:
                    if rho <= 13919.93:
                        if psi <= -0.19:
                            next.agent_mode = CraftMode.Weak_right
                            next.timer = 0
                        if psi > -0.19:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                    if rho > 13919.93:
                        next.agent_mode = CraftMode.Coc
                        next.timer = 0
