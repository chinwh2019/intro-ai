# Technical Implementation Guide

**Companion to:** REVAMP_IMPLEMENTATION_PLAN.md
**Purpose:** Concrete code patterns and implementation details
**Audience:** Developers implementing the revamp

---

## Table of Contents

1. [Core Framework Implementation](#1-core-framework-implementation)
2. [Module Interface & Plugin System](#2-module-interface--plugin-system)
3. [UI Component Examples](#3-ui-component-examples)
4. [Tutorial System Implementation](#4-tutorial-system-implementation)
5. [Challenge System Implementation](#5-challenge-system-implementation)
6. [Animation System](#6-animation-system)
7. [Event System & Communication](#7-event-system--communication)
8. [State Management](#8-state-management)
9. [Configuration & Theming](#9-configuration--theming)
10. [Best Practices & Patterns](#10-best-practices--patterns)

---

## 1. Core Framework Implementation

### 1.1 Main Application Loop

```python
# core/engine.py
import pygame
from typing import Dict, Any, Optional
from core.module_base import AIModule
from core.ui.theme import ThemeManager
from core.analytics.tracker import AnalyticsTracker

class ApplicationEngine:
    """Main application engine handling the game loop and module lifecycle"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = config.get('fps', 60)

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config['window_width'], config['window_height']),
            pygame.RESIZABLE
        )
        pygame.display.set_caption(config['title'])

        # Initialize subsystems
        self.theme = ThemeManager(config['theme'])
        self.analytics = AnalyticsTracker(config['analytics'])

        # Module management
        self.current_module: Optional[AIModule] = None
        self.module_registry: Dict[str, type] = {}

        # Event handlers
        self.event_handlers = []

    def register_module(self, name: str, module_class: type):
        """Register a module for later instantiation"""
        self.module_registry[name] = module_class
        print(f"Registered module: {name}")

    def load_module(self, name: str):
        """Load and initialize a module"""
        if name not in self.module_registry:
            raise ValueError(f"Module '{name}' not registered")

        # Cleanup current module
        if self.current_module:
            self.current_module.cleanup()

        # Instantiate new module
        module_class = self.module_registry[name]
        self.current_module = module_class(self.config, self.theme, self.analytics)
        self.current_module.initialize()

        self.analytics.track('module_loaded', {'module': name})
        print(f"Loaded module: {name}")

    def run(self):
        """Main application loop"""
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0  # Delta time in seconds

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.show_menu()
                    elif event.key == pygame.K_F1:
                        self.toggle_help()

                # Pass to current module
                if self.current_module:
                    self.current_module.handle_event(event)

            # Update
            if self.current_module:
                self.current_module.update(dt)

            # Render
            self.screen.fill(self.theme.colors['background'])
            if self.current_module:
                self.current_module.render(self.screen)

            pygame.display.flip()

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.current_module:
            self.current_module.cleanup()
        self.analytics.flush()
        pygame.quit()

    def show_menu(self):
        """Show module selection menu"""
        # Implementation for main menu
        pass

    def toggle_help(self):
        """Toggle help overlay"""
        if self.current_module:
            self.current_module.toggle_help()


def main():
    """Application entry point"""
    import json

    # Load configuration
    with open('config/app_config.json', 'r') as f:
        config = json.load(f)

    # Create and run engine
    engine = ApplicationEngine(config)

    # Register all modules
    from modules.search.module import SearchModule
    from modules.mdp.module import MDPModule
    from modules.reinforcement_learning.module import RLModule
    from modules.machine_learning.module import MLModule
    from modules.generative_ai.module import GenAIModule

    engine.register_module('search', SearchModule)
    engine.register_module('mdp', MDPModule)
    engine.register_module('rl', RLModule)
    engine.register_module('ml', MLModule)
    engine.register_module('genai', GenAIModule)

    # Start with module launcher
    engine.load_module('launcher')
    engine.run()


if __name__ == '__main__':
    main()
```

### 1.2 Configuration System

```python
# core/config.py
import json
import os
from typing import Any, Dict
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    """Application configuration"""
    window_width: int = 1280
    window_height: int = 720
    fps: int = 60
    title: str = "Introduction to AI"
    theme: str = "default"
    enable_analytics: bool = True
    analytics_file: str = "analytics.db"

    @classmethod
    def load(cls, path: str) -> 'AppConfig':
        """Load config from JSON file"""
        if not os.path.exists(path):
            # Create default config
            config = cls()
            config.save(path)
            return config

        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value"""
        return getattr(self, key, default)

    def set(self, key: str, value: Any):
        """Set config value"""
        setattr(self, key, value)


@dataclass
class ModuleConfig:
    """Module-specific configuration"""
    name: str
    display_name: str
    description: str
    icon: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> 'ModuleConfig':
        """Load module config"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
```

---

## 2. Module Interface & Plugin System

### 2.1 Base Module Class

```python
# core/module_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pygame
from core.ui.theme import ThemeManager
from core.tutorial.tutorial_manager import TutorialManager
from core.challenges.challenge_manager import ChallengeManager
from core.analytics.tracker import AnalyticsTracker

class AIModule(ABC):
    """Base class for all learning modules"""

    def __init__(
        self,
        config: Dict[str, Any],
        theme: ThemeManager,
        analytics: AnalyticsTracker
    ):
        self.config = config
        self.theme = theme
        self.analytics = analytics

        # UI state
        self.show_help_overlay = False
        self.show_settings = False

        # Subsystems
        self.tutorial_manager = None
        self.challenge_manager = None

        # Module state
        self.initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize module resources and state"""
        pass

    @abstractmethod
    def update(self, dt: float):
        """
        Update module state

        Args:
            dt: Delta time in seconds since last update
        """
        pass

    @abstractmethod
    def render(self, surface: pygame.Surface):
        """
        Render module to surface

        Args:
            surface: Pygame surface to render to
        """
        pass

    @abstractmethod
    def handle_event(self, event: pygame.event.Event):
        """
        Handle user input event

        Args:
            event: Pygame event
        """
        pass

    def cleanup(self):
        """Cleanup resources before module unload"""
        self.save_state()
        if self.tutorial_manager:
            self.tutorial_manager.cleanup()
        if self.challenge_manager:
            self.challenge_manager.cleanup()

    def get_state(self) -> Dict[str, Any]:
        """
        Get current module state for saving

        Returns:
            Dictionary containing serializable state
        """
        return {
            'initialized': self.initialized,
        }

    def set_state(self, state: Dict[str, Any]):
        """
        Restore module state from save

        Args:
            state: Previously saved state dictionary
        """
        self.initialized = state.get('initialized', False)

    def save_state(self):
        """Save current state to disk"""
        import json
        state = self.get_state()
        save_path = f"saves/{self.__class__.__name__}_state.json"
        os.makedirs('saves', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load state from disk"""
        import json
        save_path = f"saves/{self.__class__.__name__}_state.json"
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                state = json.load(f)
            self.set_state(state)

    # Common functionality

    def start_tutorial(self, tutorial_id: str):
        """Start a tutorial"""
        if self.tutorial_manager:
            self.tutorial_manager.start(tutorial_id)
            self.analytics.track('tutorial_started', {'tutorial': tutorial_id})

    def start_challenge(self, challenge_id: str):
        """Start a challenge"""
        if self.challenge_manager:
            self.challenge_manager.start(challenge_id)
            self.analytics.track('challenge_started', {'challenge': challenge_id})

    def toggle_help(self):
        """Toggle help overlay"""
        self.show_help_overlay = not self.show_help_overlay
        self.analytics.track('help_toggled', {'visible': self.show_help_overlay})

    def render_help_overlay(self, surface: pygame.Surface):
        """Render help overlay"""
        if not self.show_help_overlay:
            return

        # Semi-transparent overlay
        overlay = pygame.Surface(surface.get_size())
        overlay.set_alpha(200)
        overlay.fill(self.theme.colors['background'])
        surface.blit(overlay, (0, 0))

        # Help text
        help_text = self.get_help_text()
        y_offset = 100
        for line in help_text:
            text_surface = self.theme.fonts['body'].render(
                line, True, self.theme.colors['text']
            )
            x = (surface.get_width() - text_surface.get_width()) // 2
            surface.blit(text_surface, (x, y_offset))
            y_offset += 30

    @abstractmethod
    def get_help_text(self) -> List[str]:
        """Get help text for this module"""
        return [
            "Press ESC to return to menu",
            "Press F1 to toggle this help",
        ]
```

### 2.2 Example Module Implementation

```python
# modules/search/module.py
from core.module_base import AIModule
from modules.search.algorithms.base import SearchAlgorithm
from modules.search.algorithms.bfs import BFS
from modules.search.algorithms.dfs import DFS
from modules.search.algorithms.astar import AStar
from modules.search.environment.maze import Maze
from modules.search.visualizer.maze_viz import MazeVisualizer
from core.ui.components import Button, Slider, Panel

class SearchModule(AIModule):
    """Search algorithms learning module"""

    def initialize(self):
        """Initialize search module"""
        # Create environment
        self.maze = Maze(width=40, height=30)

        # Register algorithms
        self.algorithms = {
            'bfs': BFS(self.maze),
            'dfs': DFS(self.maze),
            'astar': AStar(self.maze),
        }
        self.current_algorithm = 'bfs'

        # Create visualizer
        self.visualizer = MazeVisualizer(
            self.maze,
            self.theme,
            cell_size=20
        )

        # Create UI components
        self.create_ui()

        # Initialize tutorial and challenge systems
        self.tutorial_manager = TutorialManager('config/tutorials/search.json')
        self.challenge_manager = ChallengeManager('config/challenges/search.json')

        # State
        self.running = False
        self.speed = 1.0
        self.step_mode = False

        self.initialized = True
        self.load_state()

    def create_ui(self):
        """Create UI components"""
        # Algorithm selection buttons
        self.algo_buttons = {}
        x_offset = 10
        for name in self.algorithms.keys():
            btn = Button(
                rect=pygame.Rect(x_offset, 10, 100, 40),
                text=name.upper(),
                theme=self.theme,
                callback=lambda n=name: self.select_algorithm(n)
            )
            self.algo_buttons[name] = btn
            x_offset += 110

        # Control buttons
        self.play_button = Button(
            rect=pygame.Rect(10, 60, 80, 40),
            text="Play",
            theme=self.theme,
            callback=self.toggle_play
        )

        self.step_button = Button(
            rect=pygame.Rect(100, 60, 80, 40),
            text="Step",
            theme=self.theme,
            callback=self.step_algorithm
        )

        self.reset_button = Button(
            rect=pygame.Rect(190, 60, 80, 40),
            text="Reset",
            theme=self.theme,
            callback=self.reset
        )

        # Speed slider
        self.speed_slider = Slider(
            rect=pygame.Rect(10, 110, 200, 30),
            min_val=0.1,
            max_val=10.0,
            initial_val=1.0,
            label="Speed",
            theme=self.theme,
            callback=self.set_speed
        )

        # Info panel
        self.info_panel = Panel(
            rect=pygame.Rect(10, 150, 260, 200),
            title="Statistics",
            theme=self.theme
        )

    def update(self, dt: float):
        """Update search module"""
        # Update UI components
        for btn in self.algo_buttons.values():
            btn.update()
        self.play_button.update()
        self.step_button.update()
        self.reset_button.update()
        self.speed_slider.update()

        # Update algorithm
        if self.running and not self.step_mode:
            algorithm = self.algorithms[self.current_algorithm]
            if not algorithm.is_complete():
                # Step algorithm based on speed
                steps = max(1, int(self.speed * dt * 60))  # Adjust for frame rate
                for _ in range(steps):
                    algorithm.step()
                    if algorithm.is_complete():
                        self.running = False
                        self.on_algorithm_complete()
                        break

        # Update tutorial if active
        if self.tutorial_manager and self.tutorial_manager.is_active():
            self.tutorial_manager.update(dt)

    def render(self, surface: pygame.Surface):
        """Render search module"""
        # Render maze
        maze_rect = pygame.Rect(280, 0, surface.get_width() - 280, surface.get_height())
        self.visualizer.render(surface, maze_rect)

        # Render algorithm state
        algorithm = self.algorithms[self.current_algorithm]
        self.visualizer.render_algorithm_state(
            surface,
            maze_rect,
            algorithm.get_visited(),
            algorithm.get_frontier(),
            algorithm.get_path()
        )

        # Render UI components
        for btn in self.algo_buttons.values():
            btn.render(surface)
        self.play_button.render(surface)
        self.step_button.render(surface)
        self.reset_button.render(surface)
        self.speed_slider.render(surface)

        # Render info panel
        self.render_info_panel(surface)

        # Render help overlay if visible
        self.render_help_overlay(surface)

        # Render tutorial overlay if active
        if self.tutorial_manager and self.tutorial_manager.is_active():
            self.tutorial_manager.render(surface)

    def render_info_panel(self, surface: pygame.Surface):
        """Render statistics panel"""
        self.info_panel.render(surface)

        # Get algorithm stats
        algorithm = self.algorithms[self.current_algorithm]
        stats = algorithm.get_statistics()

        # Render stats
        font = self.theme.fonts['body']
        y_offset = self.info_panel.rect.y + 30
        x = self.info_panel.rect.x + 10

        for key, value in stats.items():
            text = f"{key}: {value}"
            text_surface = font.render(text, True, self.theme.colors['text'])
            surface.blit(text_surface, (x, y_offset))
            y_offset += 25

    def handle_event(self, event: pygame.event.Event):
        """Handle user input"""
        # Pass to UI components
        for btn in self.algo_buttons.values():
            btn.handle_event(event)
        self.play_button.handle_event(event)
        self.step_button.handle_event(event)
        self.reset_button.handle_event(event)
        self.speed_slider.handle_event(event)

        # Handle keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.toggle_play()
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_s:
                self.step_algorithm()
            elif event.key == pygame.K_t:
                self.start_tutorial('basics')

    # Callback methods

    def select_algorithm(self, name: str):
        """Select algorithm"""
        self.current_algorithm = name
        self.reset()
        self.analytics.track('algorithm_selected', {'algorithm': name})

    def toggle_play(self):
        """Toggle play/pause"""
        self.running = not self.running
        self.play_button.text = "Pause" if self.running else "Play"
        self.analytics.track('playback_toggled', {'running': self.running})

    def step_algorithm(self):
        """Step algorithm one iteration"""
        algorithm = self.algorithms[self.current_algorithm]
        if not algorithm.is_complete():
            algorithm.step()
            self.analytics.track('algorithm_stepped')

    def reset(self):
        """Reset algorithm"""
        algorithm = self.algorithms[self.current_algorithm]
        algorithm.reset()
        self.running = False
        self.play_button.text = "Play"
        self.analytics.track('algorithm_reset')

    def set_speed(self, value: float):
        """Set playback speed"""
        self.speed = value

    def on_algorithm_complete(self):
        """Called when algorithm completes"""
        algorithm = self.algorithms[self.current_algorithm]
        stats = algorithm.get_statistics()
        self.analytics.track('algorithm_completed', {
            'algorithm': self.current_algorithm,
            'stats': stats
        })

    def get_help_text(self) -> List[str]:
        """Get help text"""
        return [
            "Search Algorithms Module",
            "",
            "Controls:",
            "  SPACE - Play/Pause",
            "  S - Step",
            "  R - Reset",
            "  T - Start Tutorial",
            "  F1 - Toggle Help",
            "  ESC - Return to Menu",
            "",
            "Click buttons to select algorithm and control execution",
            "Use slider to adjust playback speed",
        ]
```

---

## 3. UI Component Examples

### 3.1 Button Component

```python
# core/ui/components.py
import pygame
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ButtonStyle:
    """Button styling configuration"""
    bg_color: Tuple[int, int, int] = (100, 100, 100)
    hover_color: Tuple[int, int, int] = (120, 120, 120)
    pressed_color: Tuple[int, int, int] = (80, 80, 80)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    border_radius: int = 5
    border_width: int = 0
    border_color: Tuple[int, int, int] = (200, 200, 200)


class Button:
    """Interactive button component"""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        theme,
        callback: Optional[Callable] = None,
        style: Optional[ButtonStyle] = None
    ):
        self.rect = rect
        self.text = text
        self.theme = theme
        self.callback = callback
        self.style = style or ButtonStyle()

        # State
        self.hovered = False
        self.pressed = False
        self.enabled = True

        # Font
        self.font = theme.fonts['button']

        # Sound (optional)
        self.click_sound = None

    def update(self):
        """Update button state"""
        if not self.enabled:
            return

        mouse_pos = pygame.mouse.get_pos()
        self.hovered = self.rect.collidepoint(mouse_pos)

    def handle_event(self, event: pygame.event.Event):
        """Handle user input"""
        if not self.enabled:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.pressed = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.hovered:
                self.on_click()
            self.pressed = False

    def on_click(self):
        """Handle button click"""
        if self.click_sound:
            self.click_sound.play()

        if self.callback:
            self.callback()

    def render(self, surface: pygame.Surface):
        """Render button"""
        # Determine color based on state
        if not self.enabled:
            color = self.style.bg_color
        elif self.pressed:
            color = self.style.pressed_color
        elif self.hovered:
            color = self.style.hover_color
        else:
            color = self.style.bg_color

        # Draw button background
        pygame.draw.rect(
            surface,
            color,
            self.rect,
            border_radius=self.style.border_radius
        )

        # Draw border if specified
        if self.style.border_width > 0:
            pygame.draw.rect(
                surface,
                self.style.border_color,
                self.rect,
                width=self.style.border_width,
                border_radius=self.style.border_radius
            )

        # Draw text
        text_surface = self.font.render(
            self.text,
            True,
            self.style.text_color
        )
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def set_enabled(self, enabled: bool):
        """Enable or disable button"""
        self.enabled = enabled

    def set_text(self, text: str):
        """Change button text"""
        self.text = text
```

### 3.2 Slider Component

```python
class Slider:
    """Continuous value slider component"""

    def __init__(
        self,
        rect: pygame.Rect,
        min_val: float,
        max_val: float,
        initial_val: float,
        label: str,
        theme,
        callback: Optional[Callable[[float], None]] = None,
        discrete: bool = False,
        step: float = 1.0
    ):
        self.rect = rect
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.theme = theme
        self.callback = callback
        self.discrete = discrete
        self.step = step

        # State
        self.dragging = False
        self.hovered = False

        # Dimensions
        self.track_height = 4
        self.handle_radius = 8

        # Font
        self.font = theme.fonts['caption']

    def update(self):
        """Update slider state"""
        mouse_pos = pygame.mouse.get_pos()

        # Check if mouse over handle
        handle_x = self._value_to_x(self.value)
        handle_rect = pygame.Rect(
            handle_x - self.handle_radius,
            self.rect.centery - self.handle_radius,
            self.handle_radius * 2,
            self.handle_radius * 2
        )
        self.hovered = handle_rect.collidepoint(mouse_pos)

        # Update value if dragging
        if self.dragging:
            mouse_x = mouse_pos[0]
            self.value = self._x_to_value(mouse_x)

            # Snap to discrete values if needed
            if self.discrete:
                self.value = round(self.value / self.step) * self.step

            # Clamp value
            self.value = max(self.min_val, min(self.max_val, self.value))

            # Call callback
            if self.callback:
                self.callback(self.value)

    def handle_event(self, event: pygame.event.Event):
        """Handle user input"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

    def render(self, surface: pygame.Surface):
        """Render slider"""
        # Draw label
        label_surface = self.font.render(
            f"{self.label}: {self.value:.2f}",
            True,
            self.theme.colors['text']
        )
        surface.blit(label_surface, (self.rect.x, self.rect.y - 20))

        # Draw track
        track_rect = pygame.Rect(
            self.rect.x,
            self.rect.centery - self.track_height // 2,
            self.rect.width,
            self.track_height
        )
        pygame.draw.rect(
            surface,
            self.theme.colors['slider_track'],
            track_rect,
            border_radius=2
        )

        # Draw filled track
        handle_x = self._value_to_x(self.value)
        filled_rect = pygame.Rect(
            self.rect.x,
            self.rect.centery - self.track_height // 2,
            handle_x - self.rect.x,
            self.track_height
        )
        pygame.draw.rect(
            surface,
            self.theme.colors['primary'],
            filled_rect,
            border_radius=2
        )

        # Draw handle
        handle_color = self.theme.colors['primary_light'] if self.hovered or self.dragging else self.theme.colors['primary']
        pygame.draw.circle(
            surface,
            handle_color,
            (int(handle_x), self.rect.centery),
            self.handle_radius
        )

        # Draw handle border
        pygame.draw.circle(
            surface,
            self.theme.colors['text'],
            (int(handle_x), self.rect.centery),
            self.handle_radius,
            width=2
        )

    def _value_to_x(self, value: float) -> float:
        """Convert value to x coordinate"""
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + ratio * self.rect.width

    def _x_to_value(self, x: float) -> float:
        """Convert x coordinate to value"""
        ratio = (x - self.rect.x) / self.rect.width
        ratio = max(0.0, min(1.0, ratio))  # Clamp to [0, 1]
        return self.min_val + ratio * (self.max_val - self.min_val)

    def set_value(self, value: float):
        """Set slider value programmatically"""
        self.value = max(self.min_val, min(self.max_val, value))
        if self.callback:
            self.callback(self.value)
```

### 3.3 Panel Component

```python
class Panel:
    """Container panel for organizing UI elements"""

    def __init__(
        self,
        rect: pygame.Rect,
        title: str,
        theme,
        collapsible: bool = False,
        collapsed: bool = False
    ):
        self.rect = rect
        self.title = title
        self.theme = theme
        self.collapsible = collapsible
        self.collapsed = collapsed

        # Fonts
        self.title_font = theme.fonts['heading3']
        self.body_font = theme.fonts['body']

        # Dimensions
        self.title_height = 30
        self.padding = 10

        # Children
        self.children = []

    def add_child(self, child):
        """Add child component"""
        self.children.append(child)

    def update(self):
        """Update panel and children"""
        if not self.collapsed:
            for child in self.children:
                if hasattr(child, 'update'):
                    child.update()

    def handle_event(self, event: pygame.event.Event):
        """Handle events for panel and children"""
        # Handle collapse toggle
        if self.collapsible and event.type == pygame.MOUSEBUTTONDOWN:
            title_rect = pygame.Rect(
                self.rect.x,
                self.rect.y,
                self.rect.width,
                self.title_height
            )
            if title_rect.collidepoint(event.pos):
                self.collapsed = not self.collapsed
                return

        # Pass to children if not collapsed
        if not self.collapsed:
            for child in self.children:
                if hasattr(child, 'handle_event'):
                    child.handle_event(event)

    def render(self, surface: pygame.Surface):
        """Render panel"""
        # Draw background
        pygame.draw.rect(
            surface,
            self.theme.colors['panel_bg'],
            self.rect,
            border_radius=5
        )

        # Draw border
        pygame.draw.rect(
            surface,
            self.theme.colors['panel_border'],
            self.rect,
            width=2,
            border_radius=5
        )

        # Draw title bar
        title_rect = pygame.Rect(
            self.rect.x,
            self.rect.y,
            self.rect.width,
            self.title_height
        )
        pygame.draw.rect(
            surface,
            self.theme.colors['panel_title_bg'],
            title_rect,
            border_top_left_radius=5,
            border_top_right_radius=5
        )

        # Draw title text
        title_surface = self.title_font.render(
            self.title,
            True,
            self.theme.colors['panel_title_text']
        )
        title_x = self.rect.x + self.padding
        title_y = self.rect.y + (self.title_height - title_surface.get_height()) // 2
        surface.blit(title_surface, (title_x, title_y))

        # Draw collapse indicator if collapsible
        if self.collapsible:
            indicator = "▼" if not self.collapsed else "▶"
            indicator_surface = self.body_font.render(
                indicator,
                True,
                self.theme.colors['panel_title_text']
            )
            indicator_x = self.rect.right - self.padding - indicator_surface.get_width()
            surface.blit(indicator_surface, (indicator_x, title_y))

        # Render children if not collapsed
        if not self.collapsed:
            for child in self.children:
                if hasattr(child, 'render'):
                    child.render(surface)
```

---

## 4. Tutorial System Implementation

```python
# core/tutorial/tutorial_manager.py
import json
from typing import Dict, Any, List, Optional
import pygame

class TutorialStep:
    """Represents a single tutorial step"""

    def __init__(self, data: Dict[str, Any]):
        self.id = data['id']
        self.title = data['title']
        self.content = data['content']
        self.highlight_elements = data.get('highlight', [])
        self.actions = data.get('actions', [])
        self.wait_for = data.get('wait_for', None)
        self.next_type = data.get('next', 'manual')  # 'auto' or 'manual'
        self.duration = data.get('duration', 5000)  # milliseconds

    def should_auto_advance(self) -> bool:
        """Check if step should auto-advance"""
        return self.next_type == 'auto'


class TutorialManager:
    """Manages tutorial flow and display"""

    def __init__(self, tutorial_config_path: str):
        self.config_path = tutorial_config_path
        self.tutorials = self._load_tutorials()

        # State
        self.current_tutorial: Optional[str] = None
        self.current_step_index = 0
        self.active = False
        self.waiting_for_event = False
        self.step_timer = 0

        # UI
        self.overlay_alpha = 180
        self.highlight_color = (255, 255, 0)

    def _load_tutorials(self) -> Dict[str, List[TutorialStep]]:
        """Load tutorial definitions"""
        with open(self.config_path, 'r') as f:
            data = json.load(f)

        tutorials = {}
        for tutorial_data in data.get('tutorials', []):
            tutorial_id = tutorial_data['id']
            steps = [TutorialStep(step_data) for step_data in tutorial_data['steps']]
            tutorials[tutorial_id] = steps

        return tutorials

    def start(self, tutorial_id: str):
        """Start a tutorial"""
        if tutorial_id not in self.tutorials:
            print(f"Tutorial '{tutorial_id}' not found")
            return

        self.current_tutorial = tutorial_id
        self.current_step_index = 0
        self.active = True
        self.step_timer = 0
        self._execute_step_actions()

    def stop(self):
        """Stop current tutorial"""
        self.active = False
        self.current_tutorial = None
        self.current_step_index = 0

    def next_step(self):
        """Advance to next tutorial step"""
        if not self.active:
            return

        self.current_step_index += 1
        steps = self.tutorials[self.current_tutorial]

        if self.current_step_index >= len(steps):
            # Tutorial complete
            self.stop()
            self._on_tutorial_complete()
        else:
            self.step_timer = 0
            self._execute_step_actions()

    def previous_step(self):
        """Go back to previous step"""
        if not self.active:
            return

        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.step_timer = 0

    def handle_event(self, event_type: str, event_data: Any):
        """Check if event advances tutorial"""
        if not self.active or not self.waiting_for_event:
            return

        step = self._get_current_step()
        if step and step.wait_for:
            required_event = step.wait_for.get('event')
            required_value = step.wait_for.get('value')

            if event_type == required_event:
                if required_value is None or event_data == required_value:
                    self.waiting_for_event = False
                    self.next_step()

    def update(self, dt: float):
        """Update tutorial state"""
        if not self.active:
            return

        step = self._get_current_step()
        if not step:
            return

        # Check for wait condition
        if step.wait_for:
            self.waiting_for_event = True
            return

        # Auto-advance timer
        if step.should_auto_advance():
            self.step_timer += dt * 1000  # Convert to milliseconds
            if self.step_timer >= step.duration:
                self.next_step()

    def render(self, surface: pygame.Surface):
        """Render tutorial overlay"""
        if not self.active:
            return

        step = self._get_current_step()
        if not step:
            return

        # Dim everything except highlighted elements
        overlay = pygame.Surface(surface.get_size())
        overlay.set_alpha(self.overlay_alpha)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (0, 0))

        # Highlight specific elements (leave holes in overlay)
        # This requires knowing element positions from the module
        # For now, just draw highlighted borders

        # Draw instruction box
        self._render_instruction_box(surface, step)

        # Draw navigation controls
        self._render_navigation(surface)

    def _render_instruction_box(self, surface: pygame.Surface, step: TutorialStep):
        """Render instruction text box"""
        # Box dimensions
        box_width = 600
        box_height = 200
        box_x = (surface.get_width() - box_width) // 2
        box_y = surface.get_height() - box_height - 50

        # Draw box background
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(surface, (40, 40, 40), box_rect, border_radius=10)
        pygame.draw.rect(surface, (100, 100, 100), box_rect, width=2, border_radius=10)

        # Draw title
        from pygame import font
        title_font = font.Font(None, 32)
        title_surface = title_font.render(step.title, True, (255, 255, 255))
        title_x = box_x + 20
        title_y = box_y + 20
        surface.blit(title_surface, (title_x, title_y))

        # Draw content
        content_font = font.Font(None, 24)
        y_offset = title_y + 50
        for line in step.content.split('\n'):
            content_surface = content_font.render(line, True, (200, 200, 200))
            surface.blit(content_surface, (box_x + 20, y_offset))
            y_offset += 30

    def _render_navigation(self, surface: pygame.Surface):
        """Render navigation controls"""
        # Draw "Next", "Previous", "Skip" buttons
        # Implementation depends on button component
        pass

    def _execute_step_actions(self):
        """Execute actions for current step"""
        step = self._get_current_step()
        if not step:
            return

        for action in step.actions:
            action_type = action.get('type')
            # Dispatch action to module
            # This requires an event system
            pass

    def _get_current_step(self) -> Optional[TutorialStep]:
        """Get current tutorial step"""
        if not self.active or not self.current_tutorial:
            return None

        steps = self.tutorials[self.current_tutorial]
        if self.current_step_index < len(steps):
            return steps[self.current_step_index]
        return None

    def _on_tutorial_complete(self):
        """Called when tutorial completes"""
        print(f"Tutorial '{self.current_tutorial}' completed!")
        # Could trigger achievement, analytics event, etc.

    def is_active(self) -> bool:
        """Check if tutorial is active"""
        return self.active

    def cleanup(self):
        """Cleanup resources"""
        self.stop()
```

---

## 5. Challenge System Implementation

```python
# core/challenges/challenge_manager.py
import json
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

class ChallengeStatus(Enum):
    """Challenge completion status"""
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Challenge:
    """Represents a single challenge"""

    def __init__(self, data: Dict[str, Any]):
        self.id = data['id']
        self.title = data['title']
        self.description = data['description']
        self.difficulty = data.get('difficulty', 'medium')
        self.learning_objectives = data.get('learning_objectives', [])
        self.setup = data.get('setup', {})
        self.tasks = data.get('tasks', [])
        self.hints = data.get('hints', [])
        self.max_points = sum(task.get('points', 0) for task in self.tasks)

        # State
        self.status = ChallengeStatus.AVAILABLE
        self.current_points = 0
        self.completed_tasks = set()
        self.unlocked_hints = set()
        self.time_elapsed = 0

    def check_task_completion(self, task_id: str, metrics: Dict[str, Any]) -> bool:
        """Check if task is completed based on metrics"""
        task = next((t for t in self.tasks if t['id'] == task_id), None)
        if not task:
            return False

        criteria = task.get('success_criteria', {})

        # Check all criteria
        for key, condition in criteria.items():
            if key not in metrics:
                return False

            value = metrics[key]

            # Handle different condition types
            if isinstance(condition, bool):
                if value != condition:
                    return False
            elif isinstance(condition, dict):
                # Range conditions
                if 'min' in condition and value < condition['min']:
                    return False
                if 'max' in condition and value > condition['max']:
                    return False
                if 'optimal' in condition and condition['optimal']:
                    # Check if value is optimal (requires additional logic)
                    pass

        return True

    def complete_task(self, task_id: str):
        """Mark task as completed"""
        task = next((t for t in self.tasks if t['id'] == task_id), None)
        if task and task_id not in self.completed_tasks:
            self.completed_tasks.add(task_id)
            self.current_points += task.get('points', 0)

            # Check if all tasks completed
            if len(self.completed_tasks) == len(self.tasks):
                self.status = ChallengeStatus.COMPLETED

    def unlock_hint(self, hint_index: int):
        """Unlock a hint"""
        if hint_index < len(self.hints):
            self.unlocked_hints.add(hint_index)

    def get_available_hints(self) -> List[str]:
        """Get hints that can be unlocked based on time"""
        available = []
        for i, hint in enumerate(self.hints):
            unlock_time = hint.get('unlocks_after', 0)
            if self.time_elapsed >= unlock_time:
                available.append(hint['text'])
        return available


class ChallengeManager:
    """Manages challenges and progression"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.challenges = self._load_challenges()

        # State
        self.current_challenge: Optional[Challenge] = None
        self.completed_challenges = set()

        # Callbacks
        self.on_challenge_started: Optional[Callable] = None
        self.on_challenge_completed: Optional[Callable] = None
        self.on_task_completed: Optional[Callable] = None

    def _load_challenges(self) -> Dict[str, Challenge]:
        """Load challenge definitions"""
        with open(self.config_path, 'r') as f:
            data = json.load(f)

        challenges = {}
        for challenge_data in data.get('challenges', []):
            challenge = Challenge(challenge_data)
            challenges[challenge.id] = challenge

        return challenges

    def start(self, challenge_id: str):
        """Start a challenge"""
        if challenge_id not in self.challenges:
            print(f"Challenge '{challenge_id}' not found")
            return False

        challenge = self.challenges[challenge_id]
        if challenge.status == ChallengeStatus.LOCKED:
            print(f"Challenge '{challenge_id}' is locked")
            return False

        self.current_challenge = challenge
        challenge.status = ChallengeStatus.IN_PROGRESS
        challenge.time_elapsed = 0

        if self.on_challenge_started:
            self.on_challenge_started(challenge)

        return True

    def stop(self):
        """Stop current challenge"""
        if self.current_challenge:
            self.current_challenge.status = ChallengeStatus.AVAILABLE
            self.current_challenge = None

    def update(self, dt: float, metrics: Dict[str, Any]):
        """Update challenge state"""
        if not self.current_challenge:
            return

        # Update elapsed time
        self.current_challenge.time_elapsed += dt

        # Check task completion
        for task in self.current_challenge.tasks:
            task_id = task['id']
            if task_id not in self.current_challenge.completed_tasks:
                if self.current_challenge.check_task_completion(task_id, metrics):
                    self.current_challenge.complete_task(task_id)

                    if self.on_task_completed:
                        self.on_task_completed(task_id, task)

        # Check challenge completion
        if self.current_challenge.status == ChallengeStatus.COMPLETED:
            self._on_challenge_complete()

    def _on_challenge_complete(self):
        """Handle challenge completion"""
        if not self.current_challenge:
            return

        self.completed_challenges.add(self.current_challenge.id)

        if self.on_challenge_completed:
            self.on_challenge_completed(self.current_challenge)

        # Could unlock next challenge, award achievement, etc.

    def get_challenge_list(self) -> List[Dict[str, Any]]:
        """Get list of all challenges with status"""
        return [
            {
                'id': challenge.id,
                'title': challenge.title,
                'difficulty': challenge.difficulty,
                'status': challenge.status.value,
                'points': challenge.current_points,
                'max_points': challenge.max_points,
            }
            for challenge in self.challenges.values()
        ]

    def cleanup(self):
        """Cleanup resources"""
        self.stop()
```

---

## 6. Animation System

```python
# core/ui/animations.py
from typing import Callable, Optional
from enum import Enum
import math

class EasingFunction(Enum):
    """Common easing functions"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"


class Animation:
    """Represents a single animation"""

    def __init__(
        self,
        duration: float,
        start_value: float,
        end_value: float,
        easing: EasingFunction = EasingFunction.LINEAR,
        on_update: Optional[Callable[[float], None]] = None,
        on_complete: Optional[Callable[[], None]] = None
    ):
        self.duration = duration
        self.start_value = start_value
        self.end_value = end_value
        self.easing = easing
        self.on_update = on_update
        self.on_complete = on_complete

        # State
        self.elapsed = 0.0
        self.current_value = start_value
        self.completed = False

    def update(self, dt: float):
        """Update animation"""
        if self.completed:
            return

        self.elapsed += dt

        if self.elapsed >= self.duration:
            # Animation complete
            self.elapsed = self.duration
            self.current_value = self.end_value
            self.completed = True

            if self.on_update:
                self.on_update(self.current_value)

            if self.on_complete:
                self.on_complete()
        else:
            # Calculate current value
            t = self.elapsed / self.duration
            t = self._apply_easing(t)

            self.current_value = self.start_value + (self.end_value - self.start_value) * t

            if self.on_update:
                self.on_update(self.current_value)

    def _apply_easing(self, t: float) -> float:
        """Apply easing function to normalized time"""
        if self.easing == EasingFunction.LINEAR:
            return t
        elif self.easing == EasingFunction.EASE_IN:
            return t * t
        elif self.easing == EasingFunction.EASE_OUT:
            return t * (2 - t)
        elif self.easing == EasingFunction.EASE_IN_OUT:
            return t * t * (3 - 2 * t)
        elif self.easing == EasingFunction.BOUNCE:
            if t < 0.5:
                return 0.5 * (1 - math.cos(t * math.pi * 4))
            else:
                return 0.5 + 0.5 * (1 - math.cos((t - 0.5) * math.pi * 4))
        return t

    def reset(self):
        """Reset animation"""
        self.elapsed = 0.0
        self.current_value = self.start_value
        self.completed = False


class AnimationManager:
    """Manages multiple animations"""

    def __init__(self):
        self.animations = []

    def add(self, animation: Animation):
        """Add animation"""
        self.animations.append(animation)

    def update(self, dt: float):
        """Update all animations"""
        # Update animations
        for anim in self.animations:
            anim.update(dt)

        # Remove completed animations
        self.animations = [a for a in self.animations if not a.completed]

    def clear(self):
        """Clear all animations"""
        self.animations.clear()

    def has_active_animations(self) -> bool:
        """Check if any animations are active"""
        return len(self.animations) > 0


# Helper functions for common animations

def fade_in(
    duration: float,
    on_update: Callable[[float], None],
    on_complete: Optional[Callable] = None
) -> Animation:
    """Create fade in animation"""
    return Animation(
        duration=duration,
        start_value=0.0,
        end_value=1.0,
        easing=EasingFunction.EASE_OUT,
        on_update=on_update,
        on_complete=on_complete
    )


def fade_out(
    duration: float,
    on_update: Callable[[float], None],
    on_complete: Optional[Callable] = None
) -> Animation:
    """Create fade out animation"""
    return Animation(
        duration=duration,
        start_value=1.0,
        end_value=0.0,
        easing=EasingFunction.EASE_IN,
        on_update=on_update,
        on_complete=on_complete
    )


def slide_in(
    duration: float,
    start_pos: float,
    end_pos: float,
    on_update: Callable[[float], None],
    on_complete: Optional[Callable] = None
) -> Animation:
    """Create slide in animation"""
    return Animation(
        duration=duration,
        start_value=start_pos,
        end_value=end_pos,
        easing=EasingFunction.EASE_OUT,
        on_update=on_update,
        on_complete=on_complete
    )
```

---

## 7. Event System & Communication

```python
# core/events.py
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    """Standard event types"""
    # Module events
    MODULE_LOADED = "module_loaded"
    MODULE_UNLOADED = "module_unloaded"

    # Algorithm events
    ALGORITHM_STARTED = "algorithm_started"
    ALGORITHM_STEPPED = "algorithm_stepped"
    ALGORITHM_COMPLETED = "algorithm_completed"
    ALGORITHM_RESET = "algorithm_reset"

    # UI events
    BUTTON_CLICKED = "button_clicked"
    SLIDER_CHANGED = "slider_changed"
    PARAMETER_CHANGED = "parameter_changed"

    # Tutorial events
    TUTORIAL_STARTED = "tutorial_started"
    TUTORIAL_STEP_COMPLETED = "tutorial_step_completed"
    TUTORIAL_COMPLETED = "tutorial_completed"

    # Challenge events
    CHALLENGE_STARTED = "challenge_started"
    TASK_COMPLETED = "task_completed"
    CHALLENGE_COMPLETED = "challenge_completed"


@dataclass
class Event:
    """Represents an event"""
    type: EventType
    data: Dict[str, Any]
    timestamp: float


class EventBus:
    """Central event bus for application-wide communication"""

    def __init__(self):
        self.listeners: Dict[EventType, List[Callable]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to an event type"""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type"""
        if event_type in self.listeners:
            self.listeners[event_type].remove(callback)

    def emit(self, event_type: EventType, data: Dict[str, Any] = None):
        """Emit an event"""
        import time

        event = Event(
            type=event_type,
            data=data or {},
            timestamp=time.time()
        )

        if event_type in self.listeners:
            for callback in self.listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")

    def clear(self):
        """Clear all listeners"""
        self.listeners.clear()


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get global event bus"""
    return _event_bus
```

---

## 8. State Management

```python
# core/state_manager.py
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class StateManager:
    """Manages application and module state persistence"""

    def __init__(self, save_dir: str = "saves"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_state(self, module_name: str, state: Dict[str, Any]):
        """Save module state"""
        filepath = os.path.join(self.save_dir, f"{module_name}_state.json")

        # Add metadata
        state['_metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'version': '1.0'
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Load module state"""
        filepath = os.path.join(self.save_dir, f"{module_name}_state.json")

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            state = json.load(f)

        # Remove metadata
        state.pop('_metadata', None)

        return state

    def delete_state(self, module_name: str):
        """Delete saved state"""
        filepath = os.path.join(self.save_dir, f"{module_name}_state.json")
        if os.path.exists(filepath):
            os.remove(filepath)

    def list_saves(self) -> List[str]:
        """List all saved states"""
        saves = []
        for filename in os.listdir(self.save_dir):
            if filename.endswith('_state.json'):
                module_name = filename.replace('_state.json', '')
                saves.append(module_name)
        return saves
```

---

## 9. Configuration & Theming

```python
# core/ui/theme.py
import json
from typing import Dict, Tuple
import pygame

class ThemeManager:
    """Manages application theming"""

    def __init__(self, theme_name: str = "default"):
        self.theme_name = theme_name
        self.colors = {}
        self.fonts = {}

        self.load_theme(theme_name)
        self.load_fonts()

    def load_theme(self, theme_name: str):
        """Load theme from file"""
        theme_path = f"config/themes/{theme_name}.json"

        try:
            with open(theme_path, 'r') as f:
                theme_data = json.load(f)

            # Load colors
            for key, value in theme_data.get('colors', {}).items():
                if isinstance(value, str):
                    # Convert hex to RGB
                    self.colors[key] = self._hex_to_rgb(value)
                elif isinstance(value, list):
                    self.colors[key] = tuple(value)

        except FileNotFoundError:
            print(f"Theme '{theme_name}' not found, using defaults")
            self._load_default_colors()

    def _load_default_colors(self):
        """Load default color scheme"""
        self.colors = {
            'background': (30, 30, 46),
            'text': (248, 248, 242),
            'primary': (92, 124, 250),
            'primary_light': (116, 148, 255),
            'success': (81, 207, 102),
            'warning': (255, 192, 120),
            'danger': (255, 107, 107),
            'info': (77, 171, 247),
            'panel_bg': (40, 40, 60),
            'panel_border': (100, 100, 120),
            'panel_title_bg': (50, 50, 70),
            'panel_title_text': (248, 248, 242),
            'slider_track': (100, 100, 120),
        }

    def load_fonts(self):
        """Load fonts"""
        pygame.font.init()

        self.fonts = {
            'heading1': pygame.font.Font(None, 36),
            'heading2': pygame.font.Font(None, 30),
            'heading3': pygame.font.Font(None, 24),
            'body': pygame.font.Font(None, 18),
            'caption': pygame.font.Font(None, 14),
            'button': pygame.font.Font(None, 20),
        }

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_color(self, name: str, default: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[int, int, int]:
        """Get color by name"""
        return self.colors.get(name, default)
```

---

## 10. Best Practices & Patterns

### 10.1 Code Organization

**Principle: Separation of Concerns**
```python
# ✅ Good: Separate algorithm from visualization
class BFS:
    """BFS algorithm (pure logic, no UI)"""
    def step(self):
        # Algorithm logic only
        pass

class BFSVisualizer:
    """BFS visualization (UI only)"""
    def render(self, surface, algorithm):
        # Rendering logic only
        pass


# ❌ Bad: Algorithm and visualization mixed
class BFS:
    def step(self):
        # Algorithm logic
        # ... pygame rendering mixed in ...
```

### 10.2 Performance Optimization

**Principle: Minimize allocations in update loop**
```python
# ✅ Good: Reuse objects
class Visualizer:
    def __init__(self):
        self._temp_rect = pygame.Rect(0, 0, 0, 0)  # Reusable rect

    def render(self, surface):
        for cell in self.cells:
            self._temp_rect.x = cell.x
            self._temp_rect.y = cell.y
            pygame.draw.rect(surface, color, self._temp_rect)


# ❌ Bad: Create new objects every frame
class Visualizer:
    def render(self, surface):
        for cell in self.cells:
            rect = pygame.Rect(cell.x, cell.y, 20, 20)  # New allocation!
            pygame.draw.rect(surface, color, rect)
```

### 10.3 Error Handling

**Principle: Fail gracefully**
```python
# ✅ Good: Handle errors gracefully
def load_config(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config not found: {path}, using defaults")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in {path}: {e}")
        return get_default_config()


# ❌ Bad: Let exceptions crash the app
def load_config(path: str) -> dict:
    with open(path, 'r') as f:  # Crashes if file missing!
        return json.load(f)
```

### 10.4 Configuration

**Principle: Configuration over hardcoding**
```python
# ✅ Good: Use configuration
class Algorithm:
    def __init__(self, config):
        self.speed = config.get('speed', 1.0)
        self.color = config.get('color', (255, 0, 0))


# ❌ Bad: Hardcoded values
class Algorithm:
    def __init__(self):
        self.speed = 1.0  # Hard to change!
        self.color = (255, 0, 0)
```

### 10.5 Testing

**Principle: Test algorithm correctness**
```python
# tests/test_algorithms/test_bfs.py
import pytest
from modules.search.algorithms.bfs import BFS
from modules.search.environment.maze import Maze

def test_bfs_finds_path():
    """Test that BFS finds a path when one exists"""
    maze = Maze(width=5, height=5)
    bfs = BFS(maze)

    bfs.run()

    assert bfs.is_complete()
    assert bfs.path_found()
    assert len(bfs.get_path()) > 0


def test_bfs_optimal():
    """Test that BFS finds optimal path"""
    # Create simple maze with known optimal path
    maze = create_simple_maze()
    bfs = BFS(maze)

    bfs.run()

    optimal_length = 10  # Known optimal length
    assert len(bfs.get_path()) == optimal_length
```

---

## Summary

This technical guide provides concrete implementation patterns for the revamped Introduction to AI platform. Key takeaways:

1. **Modular Architecture**: Clear separation between modules, UI, and framework
2. **Event-Driven**: Use event bus for loose coupling
3. **Configuration-Driven**: External configs for flexibility
4. **Performance-Conscious**: Optimize hot paths, reuse objects
5. **Educational-Focused**: Tutorial and challenge systems built-in
6. **Maintainable**: Clean code, error handling, testing

Next steps:
1. Implement core framework (engine, base module, UI components)
2. Refactor one existing module as proof of concept
3. Iterate based on lessons learned
4. Scale to remaining modules

This architecture supports the ambitious vision laid out in the main implementation plan while remaining practical and achievable.
