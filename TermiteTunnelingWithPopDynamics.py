import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import random
import os
import warnings
warnings.filterwarnings('ignore')

class TermiteIndividualTunnels:
    def __init__(self, grid_size=80, food_sources=8, max_steps=200, num_termites=5):
        """        
        Parameters:
        grid_size: Size of the simulation grid
        food_sources: Number of food sources
        max_steps: Maximum simulation steps
        num_termites: Number of individual termites
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.food_sources = food_sources
        self.num_termites = num_termites


        # Initialise Quantities
        self.colony_food = 20.0  # Initial food at colony
        self.hunger = 0.01  # Food consumption rate per step per termite
        
        # Create output directory
        self.output_dir = "size160_food16_steps500_termites5"
        os.makedirs(self.output_dir, exist_ok=True)
    
        
        # Colony center
        self.cx, self.cy = grid_size // 2, grid_size // 2
        
        # Initialize grids
        self.tunnels = np.zeros((grid_size, grid_size))  # Permanent tunnels
        self.food = np.zeros((grid_size, grid_size))  # Food sources
        self.pheromones = np.zeros((grid_size, grid_size))  # Trail pheromones
        self.soil_hardness = np.ones((grid_size, grid_size))  # Soil difficulty
        
        # Initialize colony center
        colony_radius = 2
        for dx in range(-colony_radius, colony_radius + 1):
            for dy in range(-colony_radius, colony_radius + 1):
                nx, ny = self.cx + dx, self.cy + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= colony_radius:
                        self.tunnels[nx, ny] = 1.0
        
        # Create individual termites
        self.termites = []
        for i in range(num_termites):
            # Each termite starts at colony center
            termite = {
                'id': i,
                'x': self.cx + random.randint(-1, 1),
                'y': self.cy + random.randint(-1, 1),
                'direction': random.uniform(0, 2*np.pi),  # Random starting direction
                'speed': random.uniform(0.5, 1.5),  # Movement speed
                'memory': [],  # Memory of visited locations
                'food_carrying': 0,  # Whether carrying food back to colony
                'tunnel_strength': random.uniform(0.7, 1.3),  # Digging ability
                'exploration_radius': 0,  # How far from colony
                'state': 'exploring',  # 'exploring', 'returning', 'digging'
                'target_food': None,  # Target food source if found
                'branch_probability': random.uniform(0.1, 0.3),  # Probability to branch
            }
            self.termites.append(termite)
        
        # Initialize environment
        self.setup_food_sources()
        self.setup_soil_hardness()
        
        # Track statistics
        self.history = {
            'tunnel_length': [],
            'food_found': [],
            'active_termites': [],
            'branch_points': [],
            'colony_food': [],
            'population': []
        }
        
        self.step = 0
        self.branch_points = []  # Locations where tunnels branch
    
    def setup_food_sources(self):
        """Set up food sources randomly around the colony."""
        # Clear food
        self.food = np.zeros((self.grid_size, self.grid_size))
        
        # Place food sources at varying distances
        for _ in range(self.food_sources):
            # Random distance and angle from colony
            min_dist = 15
            max_dist = self.grid_size // 2-5
            dist = random.uniform(min_dist, max_dist)
            angle = random.uniform(0, 2*np.pi)
            
            x = int(self.cx + dist * np.cos(angle))
            y = int(self.cy + dist * np.sin(angle))
            
            # Ensure within bounds
            x = max(5, min(self.grid_size - 6, x))
            y = max(5, min(self.grid_size - 6, y))
            
            # Create food patch
            patch_size = random.randint(3, 6)
            food_amount = random.uniform(0.8, 1.0)
            
            for dx in range(-patch_size, patch_size + 1):
                for dy in range(-patch_size, patch_size + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= patch_size:
                            self.food[nx, ny] = max(
                                self.food[nx, ny],
                                food_amount * np.exp(-distance/patch_size)
                            )
        
        self.food = np.clip(self.food, 0, 1)
        self.initial_food = np.sum(self.food)
        
        print(f"Placed {self.food_sources} food sources around colony")
    
    def setup_soil_hardness(self):
        """Set up heterogeneous soil hardness."""
        # Base soil hardness
        self.soil_hardness = np.ones((self.grid_size, self.grid_size)) * 0.5
        
        # Add some random patches
        for _ in range(10):
            cx = random.randint(0, self.grid_size-1)
            cy = random.randint(0, self.grid_size-1)
            radius = random.randint(5, 12)
            
            for x in range(max(0, cx-radius), min(self.grid_size, cx+radius)):
                for y in range(max(0, cy-radius), min(self.grid_size, cy+radius)):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist <= radius:
                        # Create harder or softer soil patches
                        if random.random() < 0.3:
                            # Hard soil (more difficult to dig)
                            hardness = random.uniform(0.7, 0.9)
                        else:
                            # Soft soil (easier to dig)
                            hardness = random.uniform(0.2, 0.4)
                        
                        # Blend with existing hardness
                        blend = np.exp(-dist/radius)
                        self.soil_hardness[x, y] = (1-blend) * self.soil_hardness[x, y] + blend * hardness
        
        # Smooth the hardness map
        self.soil_hardness = ndimage.gaussian_filter(self.soil_hardness, sigma=2)
        self.soil_hardness = np.clip(self.soil_hardness, 0.1, 0.9)
        
        # Make soil near colony softer for initial tunnel development
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                dist_to_colony = np.sqrt((x - self.cx)**2 + (y - self.cy)**2)
                if dist_to_colony < 10:
                    softness = 1 - dist_to_colony / 10
                    self.soil_hardness[x, y] = self.soil_hardness[x, y] * (1 - softness*0.5)

        print(self.soil_hardness)
    
    def sense_environment(self, termite):
        """Termite senses its environment."""
        x, y = int(termite['x']), int(termite['y'])
        
        # Check boundaries
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return None, None, None
        
        # Sense food
        food_nearby = 0
        food_direction = None

                            

        # Check in a 3x3 area around termite
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                self.pheromones[dx, dy] = max(0, self.pheromones[dx, dy] - 0.1)  # sensing pheromone
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.food[nx, ny] > 0.3:
                        food_nearby += self.food[nx, ny]
                        if food_direction is None:
                            food_direction = np.arctan2(dy, dx)
        
        
        return food_nearby, food_direction
    
    def update_termite(self, termite):
        """Update an individual termite's state and position."""
        self.colony_food -= self.hunger  # Consume food for survival
        x, y = int(termite['x']), int(termite['y'])
        
        # Sense environment
        food_nearby,  food_direction = self.sense_environment(termite)
        
        # Update termite memory (remember visited locations)
        memory_entry = (x, y, self.step)
        termite['memory'].append(memory_entry)
        if len(termite['memory']) > 50:  # Limit memory size
            termite['memory'].pop(0)
        
        # STATE TRANSITIONS
        if termite['state'] == 'exploring':
            # Exploring termite behavior
            if food_nearby > 0.5 and not termite['food_carrying']:
                # Found food!
                termite['state'] = 'returning'
                termite['food_carrying'] = min(1.0, food_nearby)
                termite['target_food'] = (x, y)
                
                # Consume some food
                consume_amount = 1.0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            self.food[nx, ny] = max(0, self.food[nx, ny] - consume_amount)
                            self.pheromones[nx, ny] = min(1.0, self.pheromones[nx, ny] + 0.3)  # Strong pheromone on food
            
            # Dig tunnel at current position
            self.dig_tunnel(termite)
            
            # Decide movement direction
            if termite['target_food'] is not None:
                # Move toward food
                tx, ty = termite['target_food']
                dx_to_food = tx - x
                dy_to_food = ty - y
                food_direction = np.arctan2(dy_to_food, dx_to_food)

            new_direction = self.choose_exploration_direction(termite, food_direction)
            
        elif termite['state'] == 'returning':
            # Returning to colony with food
            # Move toward colony
            dx_to_colony = self.cx - x
            dy_to_colony = self.cy - y
            distance_to_colony = np.sqrt(dx_to_colony**2 + dy_to_colony**2)
            
            if distance_to_colony < 3:
                # Reached colony, deposit food
                termite['food_carrying'] = 0
                termite['state'] = 'exploring'
                # Reset to explore in new direction
                termite['direction'] = random.uniform(0, 2*np.pi)
                new_direction = termite['direction']
                self.colony_food += 1.0  # Add food to colony
            else:
                # Move toward colony
                colony_direction = np.arctan2(dy_to_colony, dx_to_colony)
                # Add some randomness
                colony_direction += random.uniform(-0.5, 0.5)
                new_direction = colony_direction
            
            # Continue digging tunnel
            self.dig_tunnel(termite)
            
        elif termite['state'] == 'digging':
            # Specialized digging state
            self.dig_tunnel(termite, digging_strength=1.5)
            

            # Move in current direction with less randomness
            new_direction = termite['direction'] + random.uniform(-0.2, 0.2)
            
            # Occasionally switch back to exploring
            if random.random() < 0.05:
                termite['state'] = 'exploring'
        
        # Update direction
        termite['direction'] = new_direction
        
        # Move termite
        speed = termite['speed']
        if termite['state'] == 'returning':
            speed *= 1.5  # Move faster when returning with food
        
        dx = np.cos(termite['direction']) * speed
        dy = np.sin(termite['direction']) * speed
        
        new_x = termite['x'] + dx
        new_y = termite['y'] + dy
        
        # Boundary checking
        new_x = max(0, min(self.grid_size - 1, new_x))
        new_y = max(0, min(self.grid_size - 1, new_y))
        
        termite['x'] = new_x
        termite['y'] = new_y
        
        # Update exploration radius
        dist_from_colony = np.sqrt((new_x - self.cx)**2 + (new_y - self.cy)**2)
        termite['exploration_radius'] = max(termite['exploration_radius'], dist_from_colony)
        
        # Leave pheromone trail
        px, py = int(new_x), int(new_y)
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            # Stronger pheromone when carrying food
            strength = 1 if termite['food_carrying'] else 0.0
            self.pheromones[px, py] = min(1.0, self.pheromones[px, py] + strength)
    
    def choose_exploration_direction(self, termite, food_direction=None):
        """Choose a direction for exploration."""
        x, y = int(termite['x']), int(termite['y'])
        
        # Default: continue in current direction with some randomness
        new_direction = termite['direction'] + random.uniform(-0.8, 0.8)
        
        # 1. Follow food scent if detected
        if food_direction is not None:
            # Bias toward food
            food_bias = 0.7
            new_direction = (1-food_bias) * new_direction + food_bias * food_direction
        
        # 2. Follow pheromone trails
        if self.pheromones[x, y] > 0.2:
            # Calculate gradient of pheromones
            if x > 0 and x < self.grid_size-1 and y > 0 and y < self.grid_size-1:
                grad_x = (self.pheromones[x+1, y] - self.pheromones[x-1, y]) / 2
                grad_y = (self.pheromones[x, y+1] - self.pheromones[x, y-1]) / 2
                
                if abs(grad_x) > 0.01 or abs(grad_y) > 0.01:
                    trail_direction = np.arctan2(grad_y, grad_x)
                    trail_bias = 0.4
                    new_direction = (1-trail_bias) * new_direction + trail_bias * trail_direction
        
        # 3. Avoid backtracking (prefer new areas)
        # Check if recently visited this area
        for mem_x, mem_y, mem_step in termite['memory'][-10:]:
            if abs(x - mem_x) < 3 and abs(y - mem_y) < 3:
                # Recently been here, change direction more
                new_direction += random.uniform(-0.1, 0.1)
                break
        
        # 4. Branching behavior
        if random.random() < termite['branch_probability']:
            # Create a branch point
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                if self.tunnels[x, y] > 0.3:
                    self.branch_points.append((x, y, self.step))
            
            # Change direction significantly
            branch_angle = random.uniform(-np.pi/2, np.pi/2)
            new_direction += branch_angle
        
        # 5. Soil hardness influence (avoid very hard soil)
        if self.soil_hardness[x, y] > 0.7:
            # Hard soil, turn away
            new_direction += random.uniform(-1.0, 1.0)
        
        # Keep direction in [0, 2π]
        new_direction = new_direction % (2*np.pi)
        
        return new_direction
    
    def dig_tunnel(self, termite, digging_strength=1.0):
        """Dig tunnel at termite's current position."""
        x, y = int(termite['x']), int(termite['y'])
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Calculate digging effectiveness
            base_digging = termite['tunnel_strength'] * digging_strength
            soil_factor = 1.0 / (self.soil_hardness[x, y] + 0.1)
            
            digging_power = base_digging * soil_factor
            
            # Dig tunnel
            self.tunnels[x, y] = min(1.0, self.tunnels[x, y] + digging_power * 0.1)
            
            # Also dig in neighboring cells (wider tunnels)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if random.random() < 0.3:
                            self.tunnels[nx, ny] = min(1.0, self.tunnels[nx, ny] + digging_power * 0.05)
    
    def update_pheromones(self):
        """Update pheromone evaporation and diffusion."""
        # Evaporation
        self.pheromones *= 0.999 # 0.92 defuault
        
        diffusion_kernel = np.array([[0.01, 0.02, 0.01],
                                     [0.02, 0.88, 0.02],
                                     [0.01, 0.02, 0.01]])
        self.pheromones = ndimage.convolve(self.pheromones, diffusion_kernel, mode='reflect')
        
        # Ensure pheromones near tunnels persist longer
        self.pheromones = np.clip(self.pheromones, 0, 1)
    
    def create_new_termites(self):
        """Create new termites at colony center periodically."""
        if len(self.termites) * self.hunger * 200 < self.colony_food and self.step % 30 == 0:
            # New termite born at colony
            new_termite = {
                'id': len(self.termites),
                'x': self.cx + random.uniform(-1, 1),
                'y': self.cy + random.uniform(-1, 1),
                'direction': random.uniform(0, 2*np.pi),
                'speed': random.uniform(0.5, 1.5),
                'memory': [],
                'food_carrying': 0,
                'tunnel_strength': random.uniform(0.7, 1.3),
                'exploration_radius': 0,
                'state': 'exploring',
                'target_food': None,
                'branch_probability': random.uniform(0.1, 0.3),
            }
            self.termites.append(new_termite)
            self.num_termites += 1
        elif len(self.termites) > 0 and self.step % 30 == 0:
            self.termites.pop()
            self.num_termites -= 1
            
    def update_tunnel_connections(self):
        """Connect nearby tunnels and smooth tunnel network."""
        # Find tunnel ends
        tunnel_mask = self.tunnels > 0.2
        dilated = ndimage.binary_dilation(tunnel_mask)
        tunnel_ends = dilated & ~tunnel_mask
        
        # Connect nearby ends NOT USED
        # for x in range(self.grid_size):
        #     for y in range(self.grid_size):
        #         if tunnel_ends[x, y]:
        #             # Check for nearby tunnels to connect to
        #             for dx in [-2, -1, 0, 1, 2]:
        #                 for dy in [-2, -1, 0, 1, 2]:
        #                     nx, ny = x + dx, y + dy
        #                     if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
        #                         if self.tunnels[nx, ny] > 0.3 and abs(dx) + abs(dy) > 1:
        #                             # Connect them
        #                             for i in range(abs(dx)+1):
        #                                 for j in range(abs(dy)+1):
        #                                     conn_x = x + np.sign(dx)*i
        #                                     conn_y = y + np.sign(dy)*j
        #                                     if 0 <= conn_x < self.grid_size and 0 <= conn_y < self.grid_size:
        #                                         self.tunnels[conn_x, conn_y] = max(
        #                                             self.tunnels[conn_x, conn_y], 0.4
        #                                         )
        
        # Smooth tunnels
        self.tunnels = ndimage.gaussian_filter(self.tunnels, sigma=0.3)
        
        # Ensure colony center is fully developed
        colony_radius = 3
        for dx in range(-colony_radius, colony_radius + 1):
            for dy in range(-colony_radius, colony_radius + 1):
                nx, ny = self.cx + dx, self.cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= colony_radius:
                        self.tunnels[nx, ny] = max(self.tunnels[nx, ny], 0.9 - dist*0.2)
        
        self.tunnels = np.clip(self.tunnels, 0, 1)
    
    def update_stats(self):
        """Update simulation statistics."""
        tunnel_length = np.sum(self.tunnels > 0.1)
        food_found = (self.initial_food - np.sum(self.food)) / self.initial_food * 100 if self.initial_food > 0 else 0
        active_termites = sum(1 for t in self.termites if t['state'] != 'returning')
        
        self.history['tunnel_length'].append(tunnel_length)
        self.history['food_found'].append(food_found)
        self.history['active_termites'].append(active_termites)
        self.history['branch_points'].append(len(self.branch_points))
        self.history['colony_food'].append(self.colony_food)
        self.history['population'].append(len(self.termites))
    
    def step_simulation(self):
        """Advance simulation by one step."""
        if self.step >= self.max_steps:
            return False
        
        # Update each termite
        for termite in self.termites:
            self.update_termite(termite)
        
        # Update environment
        self.update_pheromones()
        self.update_tunnel_connections()
        self.create_new_termites()
        self.update_stats()
        
        self.step += 1
        return True
    
    def save_frame(self, step):
        """Save current state as image."""
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Main view with tunnels and termites
        ax1 = plt.subplot(2, 3, 1)
        
        # Create base visualization
        combined = np.zeros((self.grid_size, self.grid_size, 3))
        
        # Blue: Tunnels (primary focus)
        tunnel_vis = np.clip(self.tunnels * 2, 0, 1)
        combined[:, :, 2] = tunnel_vis
        
        # Green: Food
        combined[:, :, 1] = self.food * 0.6
        
        # Add termite positions as red dots
        termite_positions = np.zeros((self.grid_size, self.grid_size))
        for termite in self.termites:
            x, y = int(termite['x']), int(termite['y'])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                termite_positions[x, y] = 1.0
        
        # Add termites as red
        combined[:, :, 0] = np.minimum(1.0, combined[:, :, 0] + termite_positions * 0.8)
        
        ax1.imshow(combined, origin='lower')
        ax1.set_title(f'Step {step}/{self.max_steps}\nIndividual Tunnel Formation', 
                     fontweight='bold', fontsize=12)
        
        # Mark colony center
        ax1.scatter(self.cy, self.cx, c='yellow', s=300, marker='*', 
                   edgecolor='black', linewidth=2, zorder=10, label='Colony')
        
        # Mark branch points
        if self.branch_points:
            for bx, by, _ in self.branch_points[-20:]:  # Recent branches only
                ax1.scatter(by, bx, c='magenta', s=50, marker='o', 
                          alpha=0.7, edgecolor='white', linewidth=1)
        
        ax1.legend(loc='upper right')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # Plot 2: Tunnel network alone
        ax2 = plt.subplot(2, 3, 2)
        
        # Create tunnel visualization with branch points highlighted
        tunnel_display = self.tunnels.copy()
        
        # Highlight branch points
        for bx, by, bstep in self.branch_points:
            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                # Make branch points brighter
                tunnel_display[bx, by] = min(1.0, tunnel_display[bx, by] * 1.5)
        
        im2 = ax2.imshow(tunnel_display, cmap='Blues', origin='lower', vmin=0, vmax=1)
        ax2.set_title(f'Tunnel Network\nBranches: {len(self.branch_points)}')
        ax2.scatter(self.cy, self.cx, c='yellow', s=100, marker='*', edgecolor='black')
        plt.colorbar(im2, ax=ax2)
        
        # Add tunnel stats
        tunnel_pixels = np.sum(self.tunnels > 0.1)
        ax2.text(0.02, 0.98, f'Tunnel pixels: {tunnel_pixels}\nActive termites: {len(self.termites)}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Termite paths and directions
        ax3 = plt.subplot(2, 3, 3)
        
        # Show termite positions with directions
        empty_bg = np.zeros((self.grid_size, self.grid_size, 3))
        empty_bg[:, :, 2] = self.tunnels * 0.3  # Light blue background for tunnels
        
        ax3.imshow(empty_bg, origin='lower')
        
        # Plot termites with direction arrows
        for termite in self.termites[:min(30, len(self.termites))]:  # Limit for clarity
            x, y = termite['x'], termite['y']
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Color by state
                if termite['state'] == 'exploring':
                    color = 'red'
                elif termite['state'] == 'returning':
                    color = 'green'
                else:  # digging
                    color = 'orange'
                
                # Plot termite position
                ax3.scatter(y, x, c=color, s=80, alpha=0.8, edgecolor='white')
                
                # Plot direction arrow
                dx = np.cos(termite['direction']) * 2
                dy = np.sin(termite['direction']) * 2
                ax3.arrow(y, x, dy, dx, head_width=0.5, head_length=0.7,
                         fc=color, ec=color, alpha=0.6)
        
        ax3.set_title('Termite Positions & Directions\nRed: Exploring, Green: Returning, Orange: Digging')
        ax3.scatter(self.cy, self.cx, c='yellow', s=150, marker='*', edgecolor='black')
        
        # Plot 4: Colony Food and population over time
        ax4 = plt.subplot(2, 3, 4)
        steps = np.arange(len(self.history['colony_food']))
        ax4.plot(steps, self.history['colony_food'], label='Colony Food', color='brown')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Colony Food', color='brown')
        ax4.tick_params(axis='y', labelcolor='brown')
        ax4_2 = ax4.twinx()
        ax4_2.plot(steps, self.history['population'], label='Population', color='blue')
        ax4_2.set_ylabel('Population', color='blue')
        ax4_2.tick_params(axis='y', labelcolor='blue')
        ax4.set_title('Colony Food & Population Over Time')
        ax4.legend(loc='upper right')


        
        # Plot 5: Pheromone Map
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.imshow(self.pheromones, cmap='Purples', origin='lower', vmin=0, vmax=1)
        ax5.set_title('Pheromone Concentration')
        ax5.scatter(self.cy, self.cx, c='yellow', s=100, marker='*', edgecolor='black')
        plt.colorbar(im5, ax=ax5)

  
        
        # Plot 6: Soil hardness map
        ax6 = plt.subplot(2, 3, 6)
        im6 = ax6.imshow(self.soil_hardness, cmap='YlOrBr', origin='lower', vmin=0, vmax=1)
        ax6.set_title('Soil Hardness\n(Darker = Harder to dig)')
        ax6.scatter(self.cy, self.cx, c='yellow', s=100, marker='*', edgecolor='black')
        plt.colorbar(im6, ax=ax6)
        
        # Overlay major tunnels
        major_tunnels = self.tunnels > 0.3
        y_coords, x_coords = np.where(major_tunnels)
        ax6.scatter(x_coords, y_coords, c='cyan', s=1, alpha=0.3)
        
        plt.suptitle(f'Individual Termite Tunnel Formation - {len(self.termites)} Termites', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, f'frame_{step:04d}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return filename
    
    def run_simulation(self, save_interval=5):
        """Run the complete simulation."""
        print(f"\nRunning simulation for {self.max_steps} steps...")
        print(f"Individual termites creating branching tunnels from colony")
        print(f"Saving frames to: {self.output_dir}")
        print("-" * 60)
        
        for step in range(self.max_steps):
            self.step_simulation()
            
            # Save frame at intervals
            if step % save_interval == 0 or step == self.max_steps - 1:
                filename = self.save_frame(step)
                current_tunnels = np.sum(self.tunnels > 0.2)
                active_explorers = sum(1 for t in self.termites if t['state'] == 'exploring')
                print(f"Step {step:3d}/{self.max_steps} | "
                      f"Termites: {len(self.termites):3d} | "
                      f"Tunnels: {current_tunnels:5d} pixels | "
                      f"Branches: {len(self.branch_points):3d}")
        
        print("\nSimulation complete!")
        
        # Create summary
        self.create_summary()
        
        return self
    
    def create_summary(self):
        """Create summary statistics."""
        print("\n" + "="*70)
        print("INDIVIDUAL TUNNEL FORMATION - SIMULATION SUMMARY")
        print("="*70)
        
        # Final statistics
        final_tunnels = np.sum(self.tunnels > 0.1)
        final_food_pct = self.history['food_found'][-1] if self.history['food_found'] else 0
        
        print(f"\nTunnel Network Results:")
        print(f"  Total tunnel pixels: {final_tunnels}")
        print(f"  Grid coverage: {final_tunnels/(self.grid_size**2)*100:.1f}%")
        print(f"  Branch points created: {len(self.branch_points)}")
        print(f"  Final termite population: {len(self.termites)}")
        
        print(f"\nForaging Results:")
        print(f"  Food sources: {self.food_sources}")
        print(f"  Food found: {final_food_pct:.1f}%")
        
        # Analyze tunnel structure
        print(f"\nTunnel Structure Analysis:")
        
        # Find main tunnels (connected to colony)
        colony_region = ndimage.binary_dilation(self.tunnels[self.cx-5:self.cx+6, self.cy-5:self.cy+6] > 0.3)
        labeled_tunnels, num_features = ndimage.label(self.tunnels > 0.3)
        
        # Count how many tunnel segments are connected to colony
        colony_label = labeled_tunnels[self.cx, self.cy]
        connected_tunnels = np.sum(labeled_tunnels == colony_label)
        
        print(f"  Tunnel segments: {num_features}")
        print(f"  Segments connected to colony: {connected_tunnels}")
        print(f"  Isolated tunnel segments: {num_features - 1 if colony_label > 0 else num_features}")
        
        # Calculate average tunnel width
        if final_tunnels > 0:
            # Use convolution to estimate width
            kernel = np.ones((3, 3))
            tunnel_density = ndimage.convolve((self.tunnels > 0.3).astype(float), kernel)
            avg_neighbors = np.mean(tunnel_density[self.tunnels > 0.3])
            print(f"  Average tunnel connectivity: {avg_neighbors:.2f} neighbors")
        
        print(f"\nSimulation completed in {self.step} steps")
        print(f"Frames saved to: {self.output_dir}")
        print("="*70)
        
        # Save final summary plot
        self.save_final_summary()
    
    def save_final_summary(self):
        """Save a final summary plot."""
        fig = plt.figure(figsize=(14, 10))
        
        # Final tunnel network
        ax1 = plt.subplot(2, 2, 1)
        
        # Create beautiful tunnel visualization
        tunnel_display = self.tunnels.copy()
        
        # Enhance branch points
        for bx, by, _ in self.branch_points:
            if 0 <= bx < self.grid_size and 0 <= by < self.grid_size:
                # Create star-like pattern for branch points
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) + abs(dy) <= 2:
                            nx, ny = bx + dx, by + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                tunnel_display[nx, ny] = min(1.0, tunnel_display[nx, ny] * 1.2)
        
        im1 = ax1.imshow(tunnel_display, cmap='Blues', origin='lower', vmin=0, vmax=1)
        ax1.set_title(f'Final Tunnel Network\n{len(self.branch_points)} branch points')
        ax1.scatter(self.cy, self.cx, c='yellow', s=200, marker='*', edgecolor='black', zorder=10)
        
        # Highlight branch points
        if self.branch_points:
            bx_coords = [bp[1] for bp in self.branch_points]
            by_coords = [bp[0] for bp in self.branch_points]
            ax1.scatter(bx_coords, by_coords, c='magenta', s=30, marker='o', 
                       alpha=0.7, edgecolor='white', zorder=5)
        
        plt.colorbar(im1, ax=ax1)
        
        # Development over time
        ax2 = plt.subplot(2, 2, 2)
        if len(self.history['tunnel_length']) > 1:
            steps = range(len(self.history['tunnel_length']))
            ax2.plot(steps, self.history['tunnel_length'], 'b-', linewidth=2, label='Tunnel Length')
            ax2.plot(steps, self.history['branch_points'], 'm-', linewidth=2, label='Branch Points')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Count')
            ax2.set_title('Development Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Food foraging
        ax3 = plt.subplot(2, 2, 3)
        if len(self.history['food_found']) > 1:
            steps = range(len(self.history['food_found']))
            ax3.plot(steps, self.history['food_found'], 'g-', linewidth=2)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Food Found (%)')
            ax3.set_title('Foraging Progress')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        
        # Termite population
        ax4 = plt.subplot(2, 2, 4)
        if len(self.history['active_termites']) > 1:
            steps = range(len(self.history['active_termites']))
            ax4.plot(steps, self.history['active_termites'], 'r-', linewidth=2)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Active Termites')
            ax4.set_title('Termite Population')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Individual Termite Tunnel Formation - Final Summary', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_file = os.path.join(self.output_dir, 'summary.png')
        plt.savefig(summary_file, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"\nSummary plot saved to: {summary_file}")

def main():
    """Main function to run the simulation."""
    print("="*80)
    print("TERMITE TUNNEL FORMATION")
    print("="*80)
    print("\nEach termite creates its own branching tunnel from the colony center!")
    print("\nKey features:")
    print("1. Individual termites with unique behaviors")
    print("2. Termites can be: Exploring, Returning (with food), or Digging")
    print("3. Natural branching behavior with branch points")
    print("4. Pheromone trail following")
    print("5. Heterogeneous soil affects digging difficulty")
    print("6. New termites are born at colony over time")
    print("\n" + "="*80)
    
    # Get parameters
    try:
        grid_size = int(input(f"Grid size (default 80): ") or "80")
        food_sources = int(input(f"Food sources (default 8): ") or "8")
        max_steps = int(input(f"Simulation steps (default 150): ") or "150")
        num_termites = int(input(f"Initial termites (default 5): ") or "5")
    except:
        grid_size, food_sources, max_steps, num_termites = 80, 8, 150, 5
    
    print(f"\nInitializing simulation...")
    print(f"  Grid: {grid_size}x{grid_size}")
    print(f"  Food sources: {food_sources}")
    print(f"  Steps: {max_steps}")
    print(f"  Initial termites: {num_termites}")
    print(f"  Colony center: ({grid_size//2}, {grid_size//2})")
    
    # Create and run simulation
    sim = TermiteIndividualTunnels(
        grid_size=grid_size,
        food_sources=food_sources,
        max_steps=max_steps,
        num_termites=num_termites
    )
    
    # Run simulation
    sim.run_simulation(save_interval=5)
    
    print(f"\nCheck '{sim.output_dir}' directory for frames and summary!")

if __name__ == "__main__":
    main()