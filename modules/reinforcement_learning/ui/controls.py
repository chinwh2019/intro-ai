"""Interactive UI controls for RL training"""

import pygame
from typing import Callable, Optional


class Slider:
    """Interactive slider for parameter adjustment"""

    def __init__(self, x: int, y: int, width: int, min_val: float, max_val: float,
                 initial_val: float, label: str, format_str: str = "{:.3f}"):
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.format_str = format_str

        self.dragging = False
        self.handle_radius = 8

        # Calculate handle position
        self._update_handle_pos()

    def _update_handle_pos(self):
        """Update handle position based on current value"""
        normalized = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_x = self.x + int(normalized * self.width)

    def _value_from_pos(self, mouse_x: int) -> float:
        """Calculate value from mouse position"""
        normalized = max(0, min(1, (mouse_x - self.x) / self.width))
        return self.min_val + normalized * (self.max_val - self.min_val)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events. Returns True if value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Check if clicked on handle
            if (self.handle_x - self.handle_radius <= mouse_x <= self.handle_x + self.handle_radius and
                self.y - self.handle_radius <= mouse_y <= self.y + self.height + self.handle_radius):
                self.dragging = True
                return False

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                self.dragging = False
                return True  # Value changed

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x, _ = event.pos
                self.value = self._value_from_pos(mouse_x)
                self._update_handle_pos()

        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        """Render slider"""
        color_track = (100, 100, 120)
        color_handle = (80, 250, 123)
        color_text = (248, 248, 242)

        # Draw track
        pygame.draw.rect(surface, color_track,
                        (self.x, self.y + self.height // 2 - 2, self.width, 4))

        # Draw handle
        pygame.draw.circle(surface, color_handle,
                          (self.handle_x, self.y + self.height // 2),
                          self.handle_radius)
        pygame.draw.circle(surface, (255, 255, 255),
                          (self.handle_x, self.y + self.height // 2),
                          self.handle_radius, 2)

        # Draw label and value
        label_text = font.render(self.label, True, color_text)
        value_text = font.render(self.format_str.format(self.value), True, color_handle)

        surface.blit(label_text, (self.x, self.y - 20))
        surface.blit(value_text, (self.x + self.width - value_text.get_width(), self.y - 20))


class Button:
    """Interactive button"""

    def __init__(self, x: int, y: int, width: int, height: int, text: str,
                 color: tuple = (98, 114, 164), hover_color: tuple = (139, 157, 216)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.active_color = (80, 250, 123)
        self.is_hovered = False
        self.is_active = False  # For showing selected state (e.g., active preset)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse events. Returns True if clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True

        return False

    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        """Render button"""
        if self.is_active:
            color = self.active_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.color

        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, 2, border_radius=5)

        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class RLParameterPanel:
    """Interactive parameter adjustment panel for RL training"""

    def __init__(self, x: int, y: int, width: int, on_apply: Optional[Callable] = None):
        self.x = x
        self.y = y
        self.width = width
        self.on_apply = on_apply

        self.font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 22)

        # Create sliders (increase gap from title)
        slider_y = y + 50  # Increased from 30 to 50 for more space
        slider_spacing = 60  # Increased from 55 to 60 for better readability

        self.sliders = {
            'learning_rate': Slider(
                x + 10, slider_y, width - 20,
                0.001, 0.1, 0.01,
                "Learning Rate (α)", "{:.4f}"
            ),
            'discount': Slider(
                x + 10, slider_y + slider_spacing, width - 20,
                0.8, 0.99, 0.95,
                "Discount (γ)", "{:.3f}"
            ),
            'epsilon_start': Slider(
                x + 10, slider_y + slider_spacing * 2, width - 20,
                0.1, 1.0, 1.0,
                "Epsilon Start", "{:.2f}"
            ),
            'epsilon_decay': Slider(
                x + 10, slider_y + slider_spacing * 3, width - 20,
                0.90, 0.999, 0.995,
                "Epsilon Decay", "{:.4f}"
            ),
            'speed': Slider(
                x + 10, slider_y + slider_spacing * 4, width - 20,
                10, 200, 50,
                "Game Speed", "{:.0f}"
            ),
        }

        # Create preset buttons (reduced space after sliders)
        button_y = slider_y + slider_spacing * 5 + 15  # Reduced from 30 to 15
        button_width = (width - 30) // 2
        button_height = 30
        button_spacing = 10

        # Preset buttons start 25px below the label (room for "Presets:" and "Active:" text)
        preset_buttons_y = button_y + 25
        self.preset_buttons = {
            'default': Button(x + 10, preset_buttons_y, button_width, button_height, "Default"),
            'fast': Button(x + 10 + button_width + button_spacing, preset_buttons_y,
                          button_width, button_height, "Fast"),
            'slow': Button(x + 10, preset_buttons_y + button_height + button_spacing,
                          button_width, button_height, "Slow"),
            'turbo': Button(x + 10 + button_width + button_spacing,
                          preset_buttons_y + button_height + button_spacing,
                          button_width, button_height, "Turbo"),
        }

        # Action buttons (calculated from preset buttons position)
        action_button_y = preset_buttons_y + (button_height + button_spacing) * 2 + 15  # Small gap after presets

        # Split into two columns for more buttons
        col_width = (width - 30) // 2

        self.action_buttons = {
            'apply': Button(x + 10, action_button_y, width - 20, button_height,
                          "Apply Settings", (0, 150, 0), (0, 200, 0)),
            'save': Button(x + 10, action_button_y + button_height + button_spacing,
                         col_width, button_height, "Save Model"),
            'load': Button(x + 10 + col_width + button_spacing,
                         action_button_y + button_height + button_spacing,
                         col_width, button_height, "Load Model"),
            'toggle_training': Button(x + 10, action_button_y + (button_height + button_spacing) * 2,
                                    width - 20, button_height, "Training Mode: ON",
                                    (139, 233, 253), (189, 253, 255)),
        }

        # Track training mode for button label
        self.training_mode = True

        # Track active preset
        self.active_preset = 'default'

    def set_parameters(self, learning_rate: float, discount: float,
                      epsilon_start: float, epsilon_decay: float, speed: float):
        """Set slider values"""
        self.sliders['learning_rate'].value = learning_rate
        self.sliders['discount'].value = discount
        self.sliders['epsilon_start'].value = epsilon_start
        self.sliders['epsilon_decay'].value = epsilon_decay
        self.sliders['speed'].value = speed

        # Update handle positions
        for slider in self.sliders.values():
            slider._update_handle_pos()

    def get_parameters(self) -> dict:
        """Get current parameter values"""
        return {
            'learning_rate': self.sliders['learning_rate'].value,
            'discount': self.sliders['discount'].value,
            'epsilon_start': self.sliders['epsilon_start'].value,
            'epsilon_decay': self.sliders['epsilon_decay'].value,
            'speed': self.sliders['speed'].value,
        }

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle events. Returns action string if button clicked:
        - 'apply': Apply settings
        - 'save': Save model
        - 'load': Load model
        - 'toggle_training': Toggle training mode
        - 'preset_X': Load preset X
        """
        # Handle sliders
        for slider in self.sliders.values():
            slider.handle_event(event)

        # Handle preset buttons
        for preset_name, button in self.preset_buttons.items():
            if button.handle_event(event):
                return f'preset_{preset_name}'

        # Handle action buttons
        for action_name, button in self.action_buttons.items():
            if button.handle_event(event):
                return action_name

        return None

    def set_training_mode(self, is_training: bool):
        """Update training mode button text"""
        self.training_mode = is_training
        mode_text = "ON" if is_training else "OFF (Inference)"
        self.action_buttons['toggle_training'].text = f"Training: {mode_text}"

    def set_active_preset(self, preset_name: str):
        """Set which preset is currently active (for visual feedback)"""
        self.active_preset = preset_name
        # Update button active states
        for name, button in self.preset_buttons.items():
            button.is_active = (name == preset_name)

    def render(self, surface: pygame.Surface):
        """Render parameter panel"""
        # Background (adjusted height for tighter spacing)
        panel_rect = pygame.Rect(self.x, self.y, self.width, 600)  # Reduced from 640 back to 600
        pygame.draw.rect(surface, (40, 42, 54), panel_rect, border_radius=10)
        pygame.draw.rect(surface, (68, 71, 90), panel_rect, 2, border_radius=10)

        # Title (with more space)
        title = self.title_font.render("Training Settings", True, (248, 248, 242))
        surface.blit(title, (self.x + 10, self.y + 10))  # Increased from y+5 to y+10

        # Render sliders
        for slider in self.sliders.values():
            slider.render(surface, self.font)

        # Render preset buttons section
        # Buttons are positioned at button_y, label should be just above with minimal gap
        presets_label_y = self.y + 50 + 60 * 5 + 15  # Match button_y calculation
        presets_label = self.font.render("Presets:", True, (248, 248, 242))
        surface.blit(presets_label, (self.x + 10, presets_label_y))

        # Show active preset indicator
        active_preset_label = self.font.render(f"Active: {self.active_preset.title()}", True, (80, 250, 123))
        surface.blit(active_preset_label, (self.x + self.width - active_preset_label.get_width() - 10, presets_label_y))

        # Buttons appear 25px below label (just enough room for text)
        for button in self.preset_buttons.values():
            button.render(surface, self.font)

        # Render action buttons
        for button in self.action_buttons.values():
            button.render(surface, self.font)
