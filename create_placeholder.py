#!/usr/bin/env python3
"""Create a simple placeholder image for Gauss portrait."""
try:
    from PIL import Image, ImageDraw, ImageFont
    
    img = Image.new('RGB', (400, 500), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple portrait frame
    draw.rectangle([50, 50, 350, 450], outline='gray', width=2)
    draw.ellipse([125, 80, 275, 230], outline='gray', width=2)
    
    # Add text
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((200, 280), 'Gauss Portrait', fill='black', anchor='mm', font=font)
    draw.text((200, 310), '(Placeholder)', fill='gray', anchor='mm', font=font)
    
    img.save('figures/gauss_portrait.png')
    print('Created placeholder image: figures/gauss_portrait.png')
except ImportError:
    print("PIL not available, creating minimal placeholder")
    # Create a minimal PPM file
    with open('figures/gauss_portrait.ppm', 'w') as f:
        f.write('P3\n400 500\n255\n')
        for y in range(500):
            for x in range(400):
                if 50 <= x <= 350 and 50 <= y <= 450:
                    f.write('200 200 200 ')
                else:
                    f.write('255 255 255 ')
            f.write('\n')
    print('Created PPM placeholder: figures/gauss_portrait.ppm')