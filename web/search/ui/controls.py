"""
Interactive UI controls for MDP parameters
Sliders and buttons for real-time parameter adjustment
"""

import pygame
from typing import Callable, Optional, Tuple


class Slider:
    """Interactive slider for continuous parameter adjustment"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        min_val: float,
        max_val: float,
        current_val: float,
        label: str,
        callback: Optional[Callable[[float], None]] = None
    ):
        self.x = x
        self.y = y
        self.width = width
        self.min_val = min_val
        self.max_val = max_val
        self.current_val = current_val
        self.label = label
        self.callback = callback

        # Visual properties
        self.track_height = 6
        self.handle_radius = 10
        self.dragging = False
        self.hovered = False

        # Colors
        self.track_color = (100, 100, 120)
        self.filled_track_color = (92, 124, 250)
        self.handle_color = (139, 233, 253)
        self.handle_hover_color = (189, 253, 255)
        self.text_color = (248, 248, 242)

        # Fonts
        self.font = pygame.font.Font(None, 20)
        self.value_font = pygame.font.Font(None, 18)

    def get_handle_x(self) -> int:
        """Get x position of handle based on current value"""
        ratio = (self.current_val - self.min_val) / (self.max_val - self.min_val)
        return int(self.x + ratio * self.width)

    def get_value_from_x(self, mouse_x: int) -> float:
        """Convert mouse x position to value"""
        ratio = (mouse_x - self.x) / self.width
        ratio = max(0.0, min(1.0, ratio))  # Clamp to [0, 1]
        return self.min_val + ratio * (self.max_val - self.min_val)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle mouse events

        Returns:
            True if value changed
        """
        changed = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if clicked on handle
            handle_x = self.get_handle_x()
            handle_rect = pygame.Rect(
                handle_x - self.handle_radius,
                self.y - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2
            )

            if handle_rect.collidepoint(event.pos):
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                # Call callback on release
                if self.callback:
                    self.callback(self.current_val)

        elif event.type == pygame.MOUSEMOTION:
            # Check hover
            handle_x = self.get_handle_x()
            handle_rect = pygame.Rect(
                handle_x - self.handle_radius,
                self.y - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2
            )
            self.hovered = handle_rect.collidepoint(event.pos)

            # Update value if dragging
            if self.dragging:
                new_val = self.get_value_from_x(event.pos[0])
                if abs(new_val - self.current_val) > 0.001:  # Threshold to avoid micro-changes
                    self.current_val = new_val
                    changed = True

        return changed

    def set_value(self, value: float):
        """Set slider value programmatically"""
        self.current_val = max(self.min_val, min(self.max_val, value))

    def draw(self, surface: pygame.Surface):
        """Draw the slider"""
        # Draw label
        label_text = self.font.render(self.label, True, self.text_color)
        surface.blit(label_text, (self.x, self.y - 25))

        # Draw value
        if self.label == "Discount (γ)":
            value_str = f"{self.current_val:.2f}"
        elif self.label == "Noise":
            value_str = f"{self.current_val:.2f}"
        else:  # Living reward
            value_str = f"{self.current_val:.3f}"

        value_text = self.value_font.render(value_str, True, self.text_color)
        surface.blit(value_text, (self.x + self.width + 10, self.y - 8))

        # Draw track (background)
        track_rect = pygame.Rect(
            self.x,
            self.y - self.track_height // 2,
            self.width,
            self.track_height
        )
        pygame.draw.rect(surface, self.track_color, track_rect, border_radius=3)

        # Draw filled track (up to handle)
        handle_x = self.get_handle_x()
        filled_width = handle_x - self.x
        filled_rect = pygame.Rect(
            self.x,
            self.y - self.track_height // 2,
            filled_width,
            self.track_height
        )
        pygame.draw.rect(surface, self.filled_track_color, filled_rect, border_radius=3)

        # Draw handle
        handle_color = self.handle_hover_color if (self.hovered or self.dragging) else self.handle_color
        pygame.draw.circle(surface, handle_color, (handle_x, self.y), self.handle_radius)

        # Draw handle border
        pygame.draw.circle(surface, (255, 255, 255), (handle_x, self.y), self.handle_radius, 2)

        # Draw min/max labels
        min_label = self.value_font.render(f"{self.min_val:.1f}", True, (150, 150, 150))
        max_label = self.value_font.render(f"{self.max_val:.1f}", True, (150, 150, 150))
        surface.blit(min_label, (self.x, self.y + 15))
        surface.blit(max_label, (self.x + self.width - max_label.get_width(), self.y + 15))


class Button:
    """Interactive button"""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
        callback: Optional[Callable[[], None]] = None,
        color: Tuple[int, int, int] = (98, 114, 164),
        hover_color: Tuple[int, int, int] = (139, 157, 216)
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.text_color = (255, 255, 255)

        # State
        self.hovered = False
        self.pressed = False

        # Font
        self.font = pygame.font.Font(None, 22)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle mouse events

        Returns:
            True if button was clicked
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.pressed = True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.pressed and self.hovered:
                self.pressed = False
                if self.callback:
                    self.callback()
                return True
            self.pressed = False

        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)

        return False

    def draw(self, surface: pygame.Surface):
        """Draw the button"""
        # Determine color
        if self.pressed:
            color = tuple(max(0, c - 40) for c in self.color)  # Darker when pressed
        elif self.hovered:
            color = self.hover_color
        else:
            color = self.color

        # Draw button background
        pygame.draw.rect(surface, color, self.rect, border_radius=6)

        # Draw border
        border_color = (255, 255, 255) if self.hovered else (200, 200, 200)
        pygame.draw.rect(surface, border_color, self.rect, width=2, border_radius=6)

        # Draw text
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class ParameterPanel:
    """Panel with sliders for adjusting MDP parameters"""

    def __init__(self, x: int, y: int, width: int, on_apply: Callable):
        self.x = x
        self.y = y
        self.width = width
        self.on_apply = on_apply

        # Create sliders
        slider_width = width - 80
        slider_x = x + 10

        self.discount_slider = Slider(
            x=slider_x,
            y=y + 50,  # More space from title
            width=slider_width,
            min_val=0.0,
            max_val=1.0,
            current_val=0.9,
            label="Discount (γ)"
        )

        self.noise_slider = Slider(
            x=slider_x,
            y=y + 120,  # More space between sliders
            width=slider_width,
            min_val=0.0,
            max_val=1.0,
            current_val=0.2,
            label="Noise"
        )

        self.living_reward_slider = Slider(
            x=slider_x,
            y=y + 190,  # More space
            width=slider_width,
            min_val=-1.0,
            max_val=0.0,
            current_val=-0.04,
            label="Living Reward"
        )

        # Create buttons
        button_y = y + 260  # More space for readability
        button_width = (width - 30) // 2

        self.apply_button = Button(
            x=x + 10,
            y=button_y,
            width=button_width,
            height=35,
            text="Apply Changes",
            callback=self._on_apply_click,
            color=(80, 250, 123),
            hover_color=(100, 255, 143)
        )

        self.reset_button = Button(
            x=x + 20 + button_width,
            y=button_y,
            width=button_width,
            height=35,
            text="Reset Default",
            callback=self._on_reset_click,
            color=(255, 121, 198),
            hover_color=(255, 141, 218)
        )

        # Track if parameters changed
        self.parameters_changed = False

        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def _on_apply_click(self):
        """Handle apply button click"""
        print("DEBUG: Apply button clicked!")  # Debug message
        if self.on_apply:
            params = self.get_parameters()
            print(f"DEBUG: Calling callback with params: {params}")  # Debug
            try:
                self.on_apply(params)
                self.parameters_changed = False
                print(f"\n✓ Applied new parameters:")
                print(f"  Discount: {params['discount']:.2f}")
                print(f"  Noise: {params['noise']:.2f}")
                print(f"  Living reward: {params['living_reward']:.3f}")
            except Exception as e:
                print(f"ERROR in callback: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("DEBUG: No callback registered!")

    def _on_reset_click(self):
        """Handle reset button click"""
        self.discount_slider.set_value(0.9)
        self.noise_slider.set_value(0.2)
        self.living_reward_slider.set_value(-0.04)
        self.parameters_changed = True
        print("\n✓ Reset to default parameters")

    def set_parameters(self, discount: float, noise: float, living_reward: float):
        """Set slider values"""
        self.discount_slider.set_value(discount)
        self.noise_slider.set_value(noise)
        self.living_reward_slider.set_value(living_reward)

    def get_parameters(self) -> dict:
        """Get current parameter values"""
        return {
            'discount': self.discount_slider.current_val,
            'noise': self.noise_slider.current_val,
            'living_reward': self.living_reward_slider.current_val
        }

    def handle_event(self, event: pygame.event.Event):
        """Handle events for all controls"""
        # Handle sliders
        if self.discount_slider.handle_event(event):
            self.parameters_changed = True

        if self.noise_slider.handle_event(event):
            self.parameters_changed = True

        if self.living_reward_slider.handle_event(event):
            self.parameters_changed = True

        # Handle buttons
        self.apply_button.handle_event(event)
        self.reset_button.handle_event(event)

    def draw(self, surface: pygame.Surface):
        """Draw the parameter panel"""
        # Draw panel background
        panel_rect = pygame.Rect(self.x, self.y, self.width, 300)
        pygame.draw.rect(surface, (40, 42, 54), panel_rect, border_radius=8)
        pygame.draw.rect(surface, (68, 71, 90), panel_rect, width=2, border_radius=8)

        # Draw title
        title = self.font.render("Adjust Parameters", True, (248, 248, 242))
        surface.blit(title, (self.x + 10, self.y + 10))

        # Draw sliders
        self.discount_slider.draw(surface)
        self.noise_slider.draw(surface)
        self.living_reward_slider.draw(surface)

        # Draw changed indicator
        if self.parameters_changed:
            changed_text = self.small_font.render(
                "* Parameters changed - click Apply",
                True,
                (255, 184, 108)
            )
            surface.blit(changed_text, (self.x + 10, self.y + 220))

        # Draw buttons
        self.apply_button.draw(surface)
        self.reset_button.draw(surface)


class SearchParameterPanel:
    """Parameter panel specifically for search module with 3 quick-win sliders"""

    def __init__(self, x: int, y: int, width: int, on_apply: Callable):
        self.x = x
        self.y = y
        self.width = width
        self.on_apply = on_apply

        # Create sliders
        slider_width = width - 80
        slider_x = x + 10

        # Slider 1: Animation Speed (0.1x to 10x) - more compact spacing
        self.speed_slider = Slider(
            x=slider_x,
            y=y + 40,  # Closer to title
            width=slider_width,
            min_val=0.1,
            max_val=10.0,
            current_val=1.0,
            label="Speed"
        )

        # Slider 2: Heuristic Weight (0.5 to 3.0) - for A*
        self.heuristic_weight_slider = Slider(
            x=slider_x,
            y=y + 100,  # Tighter spacing
            width=slider_width,
            min_val=0.5,
            max_val=3.0,
            current_val=1.0,
            label="Heuristic Weight"
        )

        # Slider 3: Maze Complexity (0.0 to 1.0)
        self.complexity_slider = Slider(
            x=slider_x,
            y=y + 160,  # Tighter spacing
            width=slider_width,
            min_val=0.0,
            max_val=1.0,
            current_val=0.75,
            label="Maze Complexity"
        )

        # Create buttons
        button_y = y + 220  # Tighter spacing
        button_width = (width - 30) // 2

        self.apply_button = Button(
            x=x + 10,
            y=button_y,
            width=button_width,
            height=35,
            text="Apply",
            callback=self._on_apply_click,
            color=(80, 250, 123),
            hover_color=(100, 255, 143)
        )

        self.reset_button = Button(
            x=x + 20 + button_width,
            y=button_y,
            width=button_width,
            height=35,
            text="Reset",
            callback=self._on_reset_click,
            color=(255, 121, 198),
            hover_color=(255, 141, 218)
        )

        # Track changes
        self.parameters_changed = False

        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def _on_apply_click(self):
        """Handle apply button click"""
        print("DEBUG: Apply button clicked!")
        if self.on_apply:
            params = self.get_parameters()
            print(f"DEBUG: Applying params: {params}")
            try:
                self.on_apply(params)
                self.parameters_changed = False
                print(f"\n✓ Applied parameters:")
                print(f"  Speed: {params['speed']:.1f}x")
                print(f"  Heuristic weight: {params['heuristic_weight']:.2f}")
                print(f"  Maze complexity: {params['complexity']:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("DEBUG: No callback!")

    def _on_reset_click(self):
        """Reset to defaults"""
        self.speed_slider.set_value(1.0)
        self.heuristic_weight_slider.set_value(1.0)
        self.complexity_slider.set_value(0.75)
        self.parameters_changed = True
        print("\n✓ Reset to defaults")

    def set_parameters(self, speed: float, heuristic_weight: float, complexity: float):
        """Set slider values"""
        self.speed_slider.set_value(speed)
        self.heuristic_weight_slider.set_value(heuristic_weight)
        self.complexity_slider.set_value(complexity)

    def get_parameters(self) -> dict:
        """Get current parameter values"""
        return {
            'speed': self.speed_slider.current_val,
            'heuristic_weight': self.heuristic_weight_slider.current_val,
            'complexity': self.complexity_slider.current_val
        }

    def handle_event(self, event: pygame.event.Event):
        """Handle events"""
        if self.speed_slider.handle_event(event):
            self.parameters_changed = True

        if self.heuristic_weight_slider.handle_event(event):
            self.parameters_changed = True

        if self.complexity_slider.handle_event(event):
            self.parameters_changed = True

        self.apply_button.handle_event(event)
        self.reset_button.handle_event(event)

    def draw(self, surface: pygame.Surface):
        """Draw the parameter panel"""
        # Panel background (compact to fit in sidebar)
        panel_rect = pygame.Rect(self.x, self.y, self.width, 265)  # Compact height
        pygame.draw.rect(surface, (40, 42, 54), panel_rect, border_radius=8)
        pygame.draw.rect(surface, (68, 71, 90), panel_rect, width=2, border_radius=8)

        # Title
        title = self.font.render("Quick Controls", True, (248, 248, 242))
        surface.blit(title, (self.x + 10, self.y + 10))

        # Draw sliders
        self.speed_slider.draw(surface)
        self.heuristic_weight_slider.draw(surface)
        self.complexity_slider.draw(surface)

        # Show admissibility warning for heuristic weight (next to slider, not below)
        weight = self.heuristic_weight_slider.current_val
        if weight > 1.0:
            warning = self.small_font.render(
                "⚠ Inadmissible",
                True,
                (255, 184, 108)
            )
            surface.blit(warning, (self.x + 15, self.y + 130))
        elif weight == 1.0:
            info = self.small_font.render(
                "✓ Admissible",
                True,
                (80, 250, 123)
            )
            surface.blit(info, (self.x + 15, self.y + 130))

        # Changed indicator (between last slider and buttons)
        if self.parameters_changed:
            changed = self.small_font.render(
                "* Click Apply to use new values",
                True,
                (255, 184, 108)
            )
            surface.blit(changed, (self.x + 10, self.y + 190))

        # Draw buttons
        self.apply_button.draw(surface)
        self.reset_button.draw(surface)
