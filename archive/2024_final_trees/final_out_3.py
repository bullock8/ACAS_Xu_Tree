# vOwn min:  100.0
# vOwn max:  100.0
# vInt min:  100.0
# vInt max:  100.0
# vOwn test min:  100.58169228620366
# vOwn test max:  1199.9894547134443
# vInt test min:  0.0
# vInt test max:  0.0
# Number of nodes in the tree is: 129 with ccp_alpha: 0.001
# Training score:  0.7965093333333333

# Testing score:  0.733

def predict_3(rho, theta, psi, vOwn, vInt):
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
