#        # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#        # verify
#        # This source code is licensed under the MIT license found in the
#        # LICENSE file in the root directory of this source tree.

#        import numpy as np
#        import pygame
#        from Box2D import *
#        from Box2D.b2 import *


#        class guiWorld:
#            def __init__(self, fps):
#                self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1000, 1000
#                self.TARGET_FPS = fps
#                self.PPM = 10.0  # pixels per meter
#                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
#                pygame.display.set_caption('push simulator')
#                self.clock = pygame.time.Clock()
#                self.screen_origin = b2Vec2(self.SCREEN_WIDTH / (2 * self.PPM), self.SCREEN_HEIGHT / (self.PPM * 2))
#                self.colors = {
#                    b2_staticBody: (255, 255, 255, 255),
#                    b2_dynamicBody: (163, 209, 224, 255)
#                }

#            def draw(self, bodies, bg_color=(64, 64, 64, 0)):
#                def my_draw_polygon(polygon, body, fixture):
#                    vertices = [(self.screen_origin + body.transform * v) * self.PPM for v in polygon.vertices]
#                    vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
#                    color = self.colors[body.type]
#                    if body.userData == "obs":
#                        color = (123, 128, 120, 0)
#                    if body.userData == "hand":
#                        color = (174, 136, 218, 0)

#                    pygame.draw.polygon(self.screen, color, vertices)

#                def my_draw_circle(circle, body, fixture):
#                    position = (self.screen_origin + body.transform * circle.pos) * self.PPM
#                    position = (position[0], self.SCREEN_HEIGHT - position[1])
#                    color = self.colors[body.type]
#                    if body.userData == "hand":
#                        color = (174, 136, 218, 0)
#                    pygame.draw.circle(self.screen, color, [int(x) for x in
#                                                            position], int(circle.radius * self.PPM))

#                b2PolygonShape.draw = my_draw_polygon
#                b2CircleShape.draw = my_draw_circle
#                # draw the world
#                self.screen.fill(bg_color)
#                self.clock.tick(self.TARGET_FPS)
#                for body in bodies:
#                    for fixture in body.fixtures:
#                        fixture.shape.draw(body, fixture)
#                pygame.display.flip()


#        # this is the interface to pybox2d
#        class b2WorldInterface:
#            def __init__(self, do_gui=True):
#                self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
#                self.do_gui = do_gui
#                self.TARGET_FPS = 100
#                self.TIME_STEP = 1.0 / self.TARGET_FPS
#                self.VEL_ITERS, self.POS_ITERS = 10, 10
#                self.bodies = []

#                if do_gui:
#                    self.gui_world = guiWorld(self.TARGET_FPS)
#                    # raw_input()
#                else:
#                    self.gui_world = None

#            def initialize_gui(self):
#                if self.gui_world == None:
#                    self.gui_world = guiWorld(self.TARGET_FPS)
#                self.do_gui = True

#            def stop_gui(self):
#                self.do_gui = False

#            def add_bodies(self, new_bodies):
#                """ add a single b2Body or list of b2Bodies to the world"""
#                if type(new_bodies) == list:
#                    self.bodies += new_bodies
#                else:
#                    self.bodies.append(new_bodies)

#            def step(self, show_display=True, idx=0):
#                self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)
#                if show_display and self.do_gui:
#                    self.gui_world.draw(self.bodies)


#        class end_effector:
#            def __init__(self, b2world_interface, init_pos, base, init_angle, hand_shape='rectangle', hand_size=(0.3, 1)):
#                world = b2world_interface.world
#                self.hand = world.CreateDynamicBody(position=init_pos, angle=init_angle)
#                self.hand_shape = hand_shape
#                self.hand_size = hand_size
#                # forceunit for circle and rect
#                if hand_shape == 'rectangle':
#                    rshape = b2PolygonShape(box=hand_size)
#                    self.forceunit = 30.0
#                elif hand_shape == 'circle':
#                    rshape = b2CircleShape(radius=hand_size)
#                    self.forceunit = 100.0
#                elif hand_shape == 'polygon':
#                    rshape = b2PolygonShape(vertices=hand_size)
#                else:
#                    raise Exception("%s is not a correct shape" % hand_shape)

#                self.hand.CreateFixture(
#                    shape=rshape,
#                    density=.1,
#                    friction=.1
#                )
#                self.hand.userData = "hand"

#                friction_joint = world.CreateFrictionJoint(
#                    bodyA=base,
#                    bodyB=self.hand,
#                    maxForce=2,
#                    maxTorque=2,
#                )
#                b2world_interface.add_bodies(self.hand)

#            def set_pos(self, pos, angle):
#                self.hand.position = pos
#                self.hand.angle = angle

#            def apply_wrench(self, rlvel=(0, 0), ravel=0):

#                avel = self.hand.angularVelocity
#                delta_avel = ravel - avel
#                torque = self.hand.mass * delta_avel * 30.0
#                self.hand.ApplyTorque(torque, wake=True)

#                lvel = self.hand.linearVelocity
#                delta_lvel = b2Vec2(rlvel) - b2Vec2(lvel)
#                force = self.hand.mass * delta_lvel * self.forceunit
#                self.hand.ApplyForce(force, self.hand.position, wake=True)

#            def get_state(self, verbose=False):
#                state = list(self.hand.position) + [self.hand.angle] + \
#                        list(self.hand.linearVelocity) + [self.hand.angularVelocity]
#                if verbose:
#                    print_state = ["%.3f" % x for x in state]
#                    print
#                    "position, velocity: (%s), (%s) " % \
#                    ((", ").join(print_state[:3]), (", ").join(print_state[3:]))

#                return state


#        def create_body(base, b2world_interface, body_shape, body_size, body_friction, body_density, obj_loc):
#            world = b2world_interface.world

#            link = world.CreateDynamicBody(position=obj_loc)
#            if body_shape == 'rectangle':
#                linkshape = b2PolygonShape(box=body_size)
#            elif body_shape == 'circle':
#                linkshape = b2CircleShape(radius=body_size)
#            elif body_shape == 'polygon':
#                linkshape = b2PolygonShape(vertices=body_size)
#            else:
#                raise Exception("%s is not a correct shape" % body_shape)

#            link.CreateFixture(
#                shape=linkshape,
#                density=body_density,
#                friction=body_friction,
#            )
#            friction_joint = world.CreateFrictionJoint(
#                bodyA=base,
#                bodyB=link,
#                maxForce=5,
#                maxTorque=2,
#            )

#            b2world_interface.add_bodies([link])
#            return link


#        def make_base(table_width, table_length, b2world_interface):
#            world = b2world_interface.world
#            base = world.CreateStaticBody(
#                position=(0, 0),
#                shapes=b2PolygonShape(box=(table_length, table_width)),
#            )

#            b2world_interface.add_bodies([base])
#            return base


#        def add_obstacles(b2world_interface, obsverts):
#            world = b2world_interface.world
#            obs = []
#            for verts in obsverts:
#                tmp = world.CreateStaticBody(
#                    position=(0, 0),
#                    shapes=b2PolygonShape(vertices=verts),
#                )
#                tmp.userData = "obs"
#                obs.append(tmp)

#            # add boundaries
#            x, y = sm.wbpolygon.exterior.xy
#            minx, maxx, miny, maxy = np.min(x), np.max(x), np.min(y), np.max(y)
#            centers = [(0, miny - 1), (0, maxy + 1), (minx - 1, 0), (maxx + 1, 0)]
#            boxlen = [(maxx - minx, 0.5), (maxx - minx, 0.5), (0.5, maxy - miny), (0.5, maxy - miny)]
#            for (pos, blen) in zip(centers, boxlen):
#                tmp = world.CreateStaticBody(
#                    position=pos,
#                    shapes=b2PolygonShape(box=blen),
#                )
#                obs.append(tmp)
#            b2world_interface.add_bodies(obs)


#        def run_simulation(world, body, body2, robot, robot2, xvel, yvel, \
#                           xvel2, yvel2, rtor, rtor2, simulation_steps,
#                           simulation_steps2):
#            # simulating push with fixed direction pointing from robot location to body location
#            desired_vel = np.array([xvel, yvel])
#            rvel = b2Vec2(desired_vel[0] + np.random.normal(0, 0.01), desired_vel[1] + np.random.normal(0, 0.01))

#            desired_vel2 = np.array([xvel2, yvel2])
#            rvel2 = b2Vec2(desired_vel2[0] + np.random.normal(0, 0.01), desired_vel2[1] + np.random.normal(0, 0.01))

#            tmax = np.max([simulation_steps, simulation_steps2])
#            for t in range(tmax + 100):
#                if t < simulation_steps:
#                    robot.apply_wrench(rvel, rtor)
#                if t < simulation_steps2:
#                    robot2.apply_wrench(rvel2, rtor2)
#                world.step()

#            return (list(body.position), list(body2.position))
