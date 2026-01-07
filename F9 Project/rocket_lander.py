import numpy as np
import Box2D
from Box2D.b2 import (fixtureDef, polygonShape, revoluteJointDef, distanceJointDef, contactListener)
import gymnasium as gym  # Changed from gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pygame

"""

The objective of this environment is to land a rocket on a ship.

STATE VARIABLES
The state consists of the following variables:
    - x position
    - y position
    - angle
    - first leg ground contact indicator
    - second leg ground contact indicator
    - throttle
    - engine gimbal
If VEL_STATE is set to true, the velocities are included:
    - x velocity
    - y velocity
    - angular velocity
all state variables are roughly in the range [-1, 1]
    
CONTROL INPUTS
Discrete control inputs are:
    - gimbal left
    - gimbal right
    - throttle up
    - throttle down
    - use first control thruster
    - use second control thruster
    - no action
    
Continuous control inputs are:
    - gimbal (left/right)
    - throttle (up/down)
    - control thruster (left/right)

"""

def rgb(r, g, b):
        return float(r) / 255, float(g) / 255, float(b)

CONTINUOUS = True
VEL_STATE = True  # Add velocity info to state
FPS = 60
SCALE_S = 0.35  # Temporal Scaling, lower is faster - adjust forces appropriately
INITIAL_RANDOM = 0.4  # Random scaling of initial velocity, higher is more difficult

START_HEIGHT = 1000.0
START_SPEED = 80.0

# ROCKET
MIN_THROTTLE = 0.4
GIMBAL_THRESHOLD = 0.4
MAIN_ENGINE_POWER = 1600 * SCALE_S
SIDE_ENGINE_POWER = 100 / FPS * SCALE_S

ROCKET_WIDTH = 3.66 * SCALE_S
ROCKET_HEIGHT = ROCKET_WIDTH / 3.7 * 47.9
ENGINE_HEIGHT = ROCKET_WIDTH * 0.5
ENGINE_WIDTH = ENGINE_HEIGHT * 0.7
THRUSTER_HEIGHT = ROCKET_HEIGHT * 0.78  # Lowered from 0.86
FIN_HEIGHT = ROCKET_HEIGHT * 0.88

# LEGS
LEG_LENGTH = ROCKET_WIDTH * 2.2
BASE_ANGLE = -0.27
SPRING_ANGLE = 0.27
LEG_AWAY = ROCKET_WIDTH / 2

# SHIP
SHIP_HEIGHT = ROCKET_WIDTH
SHIP_WIDTH = SHIP_HEIGHT * 40

# VIEWPORT
VIEWPORT_H = 720
VIEWPORT_W = 500
H = 0.6 * START_HEIGHT * SCALE_S
W = float(VIEWPORT_W) / VIEWPORT_H * H

# SMOKE FOR VISUALS
MAX_SMOKE_LIFETIME = 2 * FPS

MEAN = np.array([-0.034, -0.15, -0.016, 0.0024, 0.0024, 0.137,
                 - 0.02, -0.01, -0.8, 0.002])
VAR = np.sqrt(np.array([0.08, 0.33, 0.0073, 0.0023, 0.0023, 0.8,
                        0.085, 0.0088, 0.063, 0.076]))

# MORE REALISTIC COLORS
COLOR_SKY = (15, 32, 67)      # Deep midnight blue/space
COLOR_WATER = (10, 25, 45)    # Dark Atlantic water
COLOR_SHIP = (35, 35, 35)     # Dark metallic gray
COLOR_PAD = (206, 206, 2)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.water in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.lander in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.containers[0] in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.env.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        else:
            for i in range(2):
                if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class RocketLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None
        self.episode_number = 0

        self.world = Box2D.b2World()
        self.water = None
        self.lander = None
        self.engine = None
        self.ship = None
        self.legs = []

        high = np.array([1, 1, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf], dtype=np.float32)
        low = -high
        if not VEL_STATE:
            high = high[0:7]
            low = low[0:7]

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if CONTINUOUS:
            self.action_space = spaces.Box(-1.0, +1.0, (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(7)

        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.water:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.legs = []
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.throttle = 0
        self.gimbal = 0.0
        self.landed_ticks = 0
        self.stepnumber = 0
        self.smoke = []

        # self.terrainheigth = self.np_random.uniform(H / 20, H / 10)
        self.terrainheigth = H / 20
        self.shipheight = self.terrainheigth + SHIP_HEIGHT
        # ship_pos = self.np_random.uniform(0, SHIP_WIDTH / SCALE) + SHIP_WIDTH / SCALE
        ship_pos = W / 2
        self.helipad_x1 = ship_pos - SHIP_WIDTH / 2
        self.helipad_x2 = self.helipad_x1 + SHIP_WIDTH
        self.helipad_y = self.terrainheigth + SHIP_HEIGHT

        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((0, 0), (W, 0), (W, self.terrainheigth), (0, self.terrainheigth))),
                friction=0.1,
                restitution=0.0)
        )
        self.water.color1 = rgb(70, 96, 176)

        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=((self.helipad_x1, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth),
                              (self.helipad_x2, self.terrainheigth + SHIP_HEIGHT),
                              (self.helipad_x1, self.terrainheigth + SHIP_HEIGHT))),
                friction=0.5,
                restitution=0.0)
        )

        self.containers = []
        for side in [-1, 1]:
            self.containers.append(self.world.CreateStaticBody(
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=((ship_pos + side * 0.95 * SHIP_WIDTH / 2, self.helipad_y),
                                  (ship_pos + side * 0.95 * SHIP_WIDTH / 2, self.helipad_y + SHIP_HEIGHT),
                                  (ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT,
                                   self.helipad_y + SHIP_HEIGHT),
                                  (ship_pos + side * 0.95 * SHIP_WIDTH / 2 - side * SHIP_HEIGHT, self.helipad_y)
                                  )),
                    friction=0.2,
                    restitution=0.0)
            ))
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)
        for container in self.containers:
            container.color1 = (200, 20, 20)

        initial_x = W / 2 + W * self.np_random.uniform(-0.3, 0.3)
        initial_y = H * 0.95
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-ROCKET_WIDTH / 2, 0),
                                             (+ROCKET_WIDTH / 2, 0),
                                             (ROCKET_WIDTH / 2, +ROCKET_HEIGHT),
                                             (-ROCKET_WIDTH / 2, +ROCKET_HEIGHT))),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0)
        )

        self.lander.color1 = rgb(230, 230, 230)

        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY, initial_y + ROCKET_WIDTH * 0.2),
                angle=(i * BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=((0, 0), (0, LEG_LENGTH / 25), (i * LEG_LENGTH, 0), (i * LEG_LENGTH, -LEG_LENGTH / 20),
                                  (i * LEG_LENGTH / 3, -LEG_LENGTH / 7))),
                    density=1,
                    restitution=0.0,
                    friction=0.2,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * LEG_AWAY, ROCKET_WIDTH * 0.2),
                localAnchorB=(0, 0),
                enableLimit=True,
                maxMotorTorque=2500.0,
                motorSpeed=-0.05 * i,
                enableMotor=True
            )
            djd = distanceJointDef(bodyA=self.lander,
                                   bodyB=leg,
                                   anchorA=(i * LEG_AWAY, ROCKET_HEIGHT / 8),
                                   anchorB=leg.fixtures[0].body.transform * (i * LEG_LENGTH, 0),
                                   collideConnected=False,
                                   frequencyHz=0.01,
                                   dampingRatio=0.9
                                   )
            if i == 1:
                rjd.lowerAngle = -SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = + SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)

            self.legs.append(leg)

        self.lander.linearVelocity = (
            -self.np_random.uniform(0, INITIAL_RANDOM) * START_SPEED * (initial_x - W / 2) / W,
            -START_SPEED)

        self.lander.angularVelocity = (1 + INITIAL_RANDOM) * np.random.uniform(-1, 1)

        self.drawlist = self.legs + [self.water] + [self.ship] + self.containers + [self.lander]

        if CONTINUOUS:
            obs = self.step([0, 0, 0])[0]
        else:
            obs = self.step(6)[0]

        return np.array(obs, dtype=np.float32), {} # Returns observation and empty info dictionary (updated for gymnasium)

    def step(self, action):

        self.force_dir = 0

        if CONTINUOUS:
            np.clip(action, -1, 1)
            self.gimbal += action[0] * 0.15 / FPS
            self.throttle += action[1] * 0.5 / FPS
            if action[2] > 0.5:
                self.force_dir = 1
            elif action[2] < -0.5:
                self.force_dir = -1
        else:
            if action == 0:
                self.gimbal += 0.01
            elif action == 1:
                self.gimbal -= 0.01
            elif action == 2:
                self.throttle += 0.01
            elif action == 3:
                self.throttle -= 0.01
            elif action == 4:  # left
                self.force_dir = -1
            elif action == 5:  # right
                self.force_dir = 1

        self.gimbal = np.clip(self.gimbal, -GIMBAL_THRESHOLD, GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = 0 if self.throttle == 0.0 else MIN_THROTTLE + self.throttle * (1 - MIN_THROTTLE)

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (-np.sin(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power,
                 np.cos(self.lander.angle + self.gimbal) * MAIN_ENGINE_POWER * self.power)
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        # control thruster force
        force_pos_c = self.lander.position + THRUSTER_HEIGHT * np.array(
            (np.sin(self.lander.angle), np.cos(self.lander.angle)))
        force_c = (-self.force_dir * np.cos(self.lander.angle) * SIDE_ENGINE_POWER,
                   self.force_dir * np.sin(self.lander.angle) * SIDE_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse=force_c, point=force_pos_c, wake=False)

        self.world.Step(1.0 / FPS, 60, 60)

        pos = self.lander.position
        vel_l = np.array(self.lander.linearVelocity) / START_SPEED
        vel_a = self.lander.angularVelocity
        x_distance = (pos.x - W / 2) / W
        y_distance = (pos.y - self.shipheight) / (H - self.shipheight)

        angle = (self.lander.angle / np.pi) % 2
        if angle > 1:
            angle -= 2

        state = [
            2 * x_distance,
            2 * (y_distance - 0.5),
            angle,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            2 * (self.throttle - 0.5),
            (self.gimbal / GIMBAL_THRESHOLD)
        ]
        if VEL_STATE:
            state.extend([vel_l[0],
                          vel_l[1],
                          vel_a])

        # REWARD -------------------------------------------------------------------------------------------------------

        # state variables for reward
        distance = np.linalg.norm((3 * x_distance, y_distance))  # weight x position more
        speed = np.linalg.norm(vel_l)
        groundcontact = self.legs[0].ground_contact or self.legs[1].ground_contact
        brokenleg = (self.legs[0].joint.angle < 0 or self.legs[1].joint.angle > -0) and groundcontact
        outside = abs(pos.x - W / 2) > W / 2 or pos.y > H
        fuelcost = 0.1 * (0 * self.power + abs(self.force_dir)) / FPS
        landed = self.legs[0].ground_contact and self.legs[1].ground_contact and speed < 0.1
        done = False

        reward = -fuelcost

        if outside or brokenleg:
            self.game_over = True

        if self.game_over:
            done = True
        else:
            # reward shaping
            shaping = -0.5 * (distance + speed + abs(angle) ** 2)
            shaping += 0.1 * (self.legs[0].ground_contact + self.legs[1].ground_contact)
            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping
            self.prev_shaping = shaping

            if landed:
                self.landed_ticks += 1
            else:
                self.landed_ticks = 0
            if self.landed_ticks == FPS:
                reward = 1.0
                done = True

        if done:
            reward += max(-1, 0 - 2 * (speed + distance + abs(angle) + abs(vel_a)))
        elif not groundcontact:
            reward -= 0.25 / FPS

        reward = np.clip(reward, -1, 1)

        # REWARD -------------------------------------------------------------------------------------------------------

        self.stepnumber += 1

        state = (state - MEAN[:len(state)]) / VAR[:len(state)]

        # Update for gymnasium
        terminated = done
        truncated = False
        return np.array(state, dtype=np.float32), float(reward), terminated, truncated, {}

    def render(self, mode='human'):
        # Updated Colors for a more realistic "gassy" look
        COLOR_RED = (200, 20, 20)    
        COLOR_FIRE = (255, 160, 20)   
        COLOR_CORE = (255, 255, 220)  
        # Gritty grey-blue gas with alpha support
        COLOR_GAS = (170, 180, 190, 140) 

        if self.viewer is None:
            pygame.init()
            pygame.display.set_caption("Falcon 9 - Skinny Thrusters")
            self.viewer = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
            self.clock = pygame.time.Clock()
            self.stars = [(np.random.randint(0, VIEWPORT_W), np.random.randint(0, VIEWPORT_H), 
                           np.random.randint(1, 2), np.random.randint(150, 200)) for _ in range(80)]
            self.ground_smoke = [] 

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)
        
        # --- LAYER 1: Background ---
        for i in range(VIEWPORT_H):
            r, g, b = max(0, 15 - i // 50), max(0, 32 - i // 40), max(0, 67 - i // 15)
            pygame.draw.line(self.surf, (r, g, b), (0, i), (VIEWPORT_W, i))
        for x, y, size, b in self.stars:
            pygame.draw.circle(self.surf, (b, b, b), (x, y), size)

        def world_to_screen(point):
            return (int(point[0] * (VIEWPORT_W / W)), int(VIEWPORT_H - point[1] * (VIEWPORT_H / H)))

        # --- LAYER 2: Engine Fire & Ground Smoke ---
        engine_world_pos = self.lander.transform * (0, 0)
        dist_to_pad = engine_world_pos[1] - self.shipheight
        
        if self.power > 0.1:
            f_scale = self.power * np.random.uniform(0.9, 1.1)*2
            fire_poly = [(ENGINE_WIDTH * 0.4, 0), (-ENGINE_WIDTH * 0.4, 0),
                         (-ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5 * f_scale),
                         (0, -ENGINE_HEIGHT * 8 * f_scale),
                         (ENGINE_WIDTH * 1.2, -ENGINE_HEIGHT * 5 * f_scale)]
            
            cos_g, sin_g = np.cos(self.gimbal), np.sin(self.gimbal)
            def get_transformed_poly(poly):
                transformed = []
                for vx, vy in poly:
                    rx, ry = vx * cos_g - vy * sin_g, vx * sin_g + vy * cos_g
                    transformed.append(world_to_screen(self.lander.transform * (rx, ry)))
                return transformed

            pygame.draw.polygon(self.surf, COLOR_FIRE, get_transformed_poly(fire_poly))
            pygame.draw.polygon(self.surf, COLOR_CORE, get_transformed_poly([(v[0]*0.5, v[1]*0.6) for v in fire_poly]))

        # --- LAYER 3: Rocket Body & Nose Thruster Puffs ---
        # Draw Rocket Body
        # --- LAYER 4: Rocket Body & Grid Fins ---
        for fixture in self.lander.fixtures:
            pygame.draw.polygon(self.surf, (240, 240, 240), [world_to_screen(self.lander.transform * v) for v in fixture.shape.vertices])

        # Grid Fins (Now positioned ABOVE the thrusters)
        for i in (-1, 1):
            fin_poly = [
                (i * ROCKET_WIDTH * 0.4, FIN_HEIGHT + 0.4), 
                (i * ROCKET_WIDTH * 1.2, FIN_HEIGHT + 0.4),
                (i * ROCKET_WIDTH * 1.2, FIN_HEIGHT - 0.4), 
                (i * ROCKET_WIDTH * 0.4, FIN_HEIGHT - 0.4)
            ]
            pygame.draw.polygon(self.surf, COLOR_RED, [world_to_screen(self.lander.transform * v) for v in fin_poly])

        # REWRITTEN NOSE THRUSTER SECTION
        if self.force_dir != 0:
            puff_side = 1 if self.force_dir > 0 else -1
            
            # Position at the top, slightly offset from center
            gas_port_local = (puff_side * ROCKET_WIDTH * 0.4, THRUSTER_HEIGHT)
            
            # Create a skinny "jet" polygon
            # This makes a long, thin triangle/trapezoid for a more realistic pressure jet
            jet_length = np.random.uniform(1.5, 3.0)
            jet_width = 0.2
            gas_poly_local = [
                (gas_port_local[0], gas_port_local[1] + jet_width),
                (gas_port_local[0], gas_port_local[1] - jet_width),
                (gas_port_local[0] + puff_side * jet_length, gas_port_local[1] - jet_width * 0.5),
                (gas_port_local[0] + puff_side * jet_length, gas_port_local[1] + jet_width * 0.5),
            ]
            
            gas_poly_screen = [world_to_screen(self.lander.transform * v) for v in gas_poly_local]
            pygame.draw.polygon(self.surf, COLOR_GAS, gas_poly_screen)
            
            # Add a very thin white core for high pressure
            core_poly_local = [
                (gas_port_local[0], gas_port_local[1] + jet_width * 0.3),
                (gas_port_local[0] + puff_side * jet_length * 0.6, gas_port_local[1])
            ]
            pygame.draw.line(self.surf, (220, 220, 220, 180), 
                             world_to_screen(self.lander.transform * core_poly_local[0]), 
                             world_to_screen(self.lander.transform * core_poly_local[1]), 2)

        # --- LAYER 4: The Boat & Legs ---
        # (Rest of the rendering code remains the same for ship and legs...)
        for obj in self.drawlist:
            if obj == self.lander or obj in self.legs: continue
            is_pad = (obj == self.ship or obj in self.containers)
            color = COLOR_RED if is_pad else getattr(obj, 'color1', (80,80,80))
            for fixture in obj.fixtures:
                vertices = [world_to_screen(obj.transform * v) for v in fixture.shape.vertices]
                pygame.draw.polygon(self.surf, color, vertices)

        # Centered & Fixed Leg Sweep
        height_pct = self.lander.position[1] / H
        deploy_factor = np.clip((0.60 - height_pct) / 0.30, 0.0, 1.0)
        for side in [-1, 1]:
            hinge_local = (side * ROCKET_WIDTH * 0, 0.5)
            p1_world = self.lander.transform * hinge_local
            sweep_angle = 2.5 * deploy_factor
            foot_rel_x = side * np.sin(sweep_angle) * LEG_LENGTH
            foot_rel_y = np.cos(sweep_angle) * LEG_LENGTH
            p2_world = self.lander.transform * (hinge_local[0] + foot_rel_x, hinge_local[1] + foot_rel_y)
            pygame.draw.line(self.surf, COLOR_RED, world_to_screen(p1_world), world_to_screen(p2_world), 6)

        self.viewer.blit(self.surf, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.viewer is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.viewer = None


def test_thrust_dynamics():
    env = RocketLander()
    env.reset()
    
    print("Testing: Gimbal sweep, Throttle pulsing, and Side Thruster firing.")
    
    # We'll use a sine wave to make the movements smooth and visible
    for i in range(1200):
        # 1. Maintain a hovering height to watch the effects
        y_pos = H * 0.4 
        env.lander.position = (W / 2, y_pos)
        env.lander.angle = 0  # Keep rocket vertical to see plume gimbal clearly
        env.lander.linearVelocity = (0, 0)
        env.game_over = False 

        # 2. Dynamic Controls using math.sin
        # Gimbal: Swings left and right
        gimbal_input = np.sin(i * 0.05) 
        
        # Throttle: Pulses between 0.2 and 1.0 to see plume grow/shrink
        throttle_input = 0.6 + 0.4 * np.sin(i * 0.1)
        
        # Side Thrusters: Fire left for 50 frames, then right for 50 frames
        # In your code, action[2] > 0.5 is one way, < -0.5 is the other
        thruster_input = 0
        if (i // 50) % 3 == 1:
            thruster_input = 1.0  # Fire Right
        elif (i // 50) % 3 == 2:
            thruster_input = -1.0 # Fire Left

        # 3. Step with the dynamic actions
        # Action format: [gimbal, throttle, side_thruster]
        env.step([gimbal_input, throttle_input, thruster_input])
        env.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return

    print("Test Complete.")

if __name__ == "__main__":
    test_thrust_dynamics()