# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 161 with ccp_alpha: 0.001
# Training score:  0.9252196666666667

# Testing score:  0.874

def predict_0(rho, theta, psi, vOwn, vInt):
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
