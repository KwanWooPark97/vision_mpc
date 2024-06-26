import math
from typing import Optional, Union
from scipy.integrate import odeint
import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
from math import cos,sin
"""
Utility functions used for classic control environments.
"""

from typing import Optional, SupportsFloat, Tuple


def verify_number_and_cast(x: SupportsFloat) -> float:
    """Verify parameter is a single number and cast to a float."""
    try:
        x = float(x)
    except (ValueError, TypeError):
        raise ValueError(f"An option ({x}) could not be converted to a float.")
    return x


def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Tuple[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.
    Args:
      options: Options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.
    Returns:
      Tuple of the lower and upper limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high

    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) must be lower than higher bound ({high})."
        )

    return low, high

class CartPoleEnv(gym.Env):
    """
    ### Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.
    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    ### Arguments
    ```
    gym.make('CartPole-v1')
    ```
    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        #self.tau = 0.1  # seconds between state updates
        self.tau= 0.1
        self.kinematics_integrator = "euler"
        self.time=0.0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None
    def pendulum(self,state, t, u):
        # Inputs (1):
        # Forc
        force = u

        # States (4):
        # x = state[0]
        # x_dot=state[1]
        theta = state[2]
        theta_dot = state[3]

        costheta = math.cos(math.radians(theta))
        sintheta = math.sin(math.radians(theta))
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Return xdot:
        xdot = np.zeros(4)
        xdot[3] = thetaacc
        xdot[2] = state[3]
        xdot[1] = xacc
        xdot[0] = state[1]

        return xdot

    def step(self, action):
        #err_msg = f"{action!r} ({type(action)}) invalid"
        #assert self.action_space.contains(action), err_msg
        #assert self.state is not None, "Call reset before using step method."
        force = action
        x, x_dot, theta, theta_dot = self.state
        '''ts = [self.time,self.time+0.08]
        y = odeint(self.pendulum, self.state, ts, args=(force,))
        # retrieve measurements
        self.state[0]= y[-1][0]
        self.state[1]= y[-1][1]
        self.state[2]= y[-1][2]
        self.state[3]= y[-1][3]
        #force = self.force_mag if action == 1 else -self.force_mag
        self.time+=0.08'''
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = ( force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        '''Kd=10
        z_dot=x_dot
        mPend=1.0
        L=0.5
        g=9.81
        mCart=1.0
        xacc=(force - Kd*z_dot - mPend*L*theta_dot**2*math.sin(theta) + mPend*g*math.sin(theta)*math.cos(theta)) / (mCart + mPend*math.sin(theta)**2)
        thetaacc=((force - Kd*z_dot - mPend*L*theta_dot**2*math.sin(theta))*math.cos(theta)/(mCart + mPend) + g*math.sin(theta)) / (L - mPend*L*math.cos(theta)**2/(mCart + mPend))'''


        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        if theta >= 2*math.pi:
            theta=theta%(2*math.pi)
        elif theta <= -2*math.pi:
            theta=theta%(2*math.pi)
        self.state = (x, x_dot, theta, theta_dot)

        '''terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )'''

        '''if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0'''

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32)#, reward, terminated, False, {}
    def test_step(self, test,action):
        #err_msg = f"{action!r} ({type(action)}) invalid"
        #assert self.action_space.contains(action), err_msg
        #assert self.state is not None, "Call reset before using step method."
        force = action

        x, x_dot, theta, theta_dot = test
        '''ts = np.array([self.time+i*self.tau for i in range(50)])
        y = odeint(self.pendulum, test, ts, args=(force,))
        # retrieve measurements
        test[0]= y[-1][0]
        test[1]= y[-1][1]
        test[2]= y[-1][2]
        test[3]= y[-1][3]
        #force = self.force_mag if action == 1 else -self.force_mag
        self.time+=self.tau'''
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = ( force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        '''Kd=10
        z_dot=x_dot
        mPend=1.0
        L=0.5
        g=9.81
        mCart=1.0
        xacc=(force - Kd*z_dot - mPend*L*theta_dot**2*math.sin(theta) + mPend*g*math.sin(theta)*math.cos(theta)) / (mCart + mPend*math.sin(theta)**2)
        thetaacc=((force - Kd*z_dot - mPend*L*theta_dot**2*math.sin(theta))*math.cos(theta)/(mCart + mPend) + g*math.sin(theta)) / (L - mPend*L*math.cos(theta)**2/(mCart + mPend))'''


        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        if theta >= 2*math.pi:
            theta=theta%(2*math.pi)
        elif theta <= -2*math.pi:
            theta=theta%(2*math.pi)
        test = (x, x_dot, theta, theta_dot)

        '''terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )'''

        '''if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0'''

        #if self.render_mode == "human":
        #    self.render()
        return np.array(test, dtype=np.float32)#, reward, terminated, False, {}

    def test_step_1(self,state):
        # simulation time
        dt = 0.1
        Tmax = 5
        t = np.arange(0.0, Tmax, dt)
        pi=np.pi
        g = 9.8
        L = 1.0
        m = 0.5
        # Controller coefficients
        Kp_th = 50
        Kd_th = 15
        Kp_x = 3.1
        Kd_x = 4.8
        stabilizing = False

        def energy(th, dth):
            return m * dth * L * dth * L / 2 + m * g * L * (cos(th) - 1)

        def isControllable(th, dth):
            return th < pi / 9 and abs(energy(th, dth)) < 0.5

        def derivatives(state, t):
            global stabilizing
            ds = np.zeros_like(state)

            _th = state[0]
            _Y = state[1]  # th'
            _x = state[2]
            _Z = state[3]  # x'
            k = 0.08  # control gain coefficient
            if stabilizing or isControllable(_th, _Y):
                stabilizing = True
                u = Kp_th * _th + Kd_th * _Y + Kp_x * _x + Kd_x * _Z
            else:
                E = energy(_th, _Y)
                u = k * E * _Y * cos(_th)

            ds[0] = state[1]
            ds[1] = (g * sin(_th) - u * cos(_th)) / L
            ds[2] = state[3]
            ds[3] = u

            return ds

        solution = odeint(derivatives, state, t)

        return solution

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        #super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = [0,0,3.14,0]
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
