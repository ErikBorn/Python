import pygame
import sys

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sprite Movement and Dance")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

# Load or create a sprite
sprite_size = 50
sprite = pygame.Surface((sprite_size, sprite_size))
sprite.fill(BLUE)
sprite_rect = sprite.get_rect(center=(WIDTH // 2, HEIGHT // 2))

# Animation variables
dance_frames = [
    pygame.Surface((sprite_size, sprite_size)),  # Frame 1
    pygame.Surface((sprite_size, sprite_size))   # Frame 2
]
dance_frames[0].fill((0, 255, 0))  # Green for frame 1
dance_frames[1].fill((255, 0, 0))  # Red for frame 2
current_frame = 0
animation_timer = 0
animation_speed = 200  # Milliseconds per frame

# Movement variables
speed = 5

# Main game loop
def main():
    global current_frame, animation_timer

    is_dancing = False

    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get keys pressed
        keys = pygame.key.get_pressed()
        
        # Movement controls
        if not is_dancing:  # Only allow movement if not dancing
            if keys[pygame.K_UP]:
                sprite_rect.y -= speed
            if keys[pygame.K_DOWN]:
                sprite_rect.y += speed
            if keys[pygame.K_LEFT]:
                sprite_rect.x -= speed
            if keys[pygame.K_RIGHT]:
                sprite_rect.x += speed

        # Dance toggle
        is_dancing = keys[pygame.K_SPACE]

        # Update animation if dancing
        if is_dancing:
            now = pygame.time.get_ticks()
            if now - animation_timer > animation_speed:
                current_frame = (current_frame + 1) % len(dance_frames)
                animation_timer = now

        # Drawing
        screen.fill(WHITE)  # Clear the screen

        if is_dancing:
            screen.blit(dance_frames[current_frame], sprite_rect.topleft)
        else:
            screen.blit(sprite, sprite_rect.topleft)

        pygame.display.flip()  # Update the screen

        # Cap the frame rate
        clock.tick(FPS)

if __name__ == "__main__":
    main()