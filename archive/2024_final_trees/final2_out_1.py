# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 71 with ccp_alpha: 0.002
# Training score:  0.8651623333333334

# Testing score:  0.831

def predict_1(rho, theta, psi, vOwn, vInt):
    if psi <= -2.79:
        if theta <= 0.32:
            if rho <= 13513.51:
                if theta <= 0.06:
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
                    next.agent_mode = CraftMode.Strong_left
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
                            next.agent_mode = CraftMode.Strong_right
                            next.timer = 0
                        if psi > 1.46:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                if rho > 1930.5:
                    if psi <= -0.57:
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
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
            if theta > -0.51:
                if psi <= -0.25:
                    if theta <= 0.38:
                        if rho <= 4165.82:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if rho > 4165.82:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                    if theta > 0.38:
                        if rho <= 1524.08:
                            next.agent_mode = CraftMode.Strong_left
                            next.timer = 0
                        if rho > 1524.08:
                            if theta <= 2.03:
                                next.agent_mode = CraftMode.Strong_right
                                next.timer = 0
                            if theta > 2.03:
                                next.agent_mode = CraftMode.Weak_left
                                next.timer = 0
                if psi > -0.25:
                    if theta <= 1.21:
                        if theta <= -0.19:
                            if psi <= 2.54:
                                next.agent_mode = CraftMode.Strong_right
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
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
        if rho > 9042.88:
            if theta <= 0.76:
                if theta <= -1.08:
                    if rho <= 13513.51:
                        next.agent_mode = CraftMode.Coc
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
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
                    if psi > 0.57:
                        if theta <= 0.06:
                            next.agent_mode = CraftMode.Weak_left
                            next.timer = 0
                        if theta > 0.06:
                            next.agent_mode = CraftMode.Coc
                            next.timer = 0
            if theta > 0.76:
                next.agent_mode = CraftMode.Coc
                next.timer = 0
