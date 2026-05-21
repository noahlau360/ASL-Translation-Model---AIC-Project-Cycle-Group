import os
import asyncio
from dotenv import load_dotenv
from tplinkcloud import TPLinkDeviceManager

# This line finds the .env file and loads the variables into your system
load_dotenv()
KASA_USER = os.getenv("KASA_USER")
KASA_PASS = os.getenv("KASA_PASS")
DEVICE_NAME = os.getenv("DEVICE_NAME")

async def main():
    device_manager = TPLinkDeviceManager(KASA_USER, KASA_PASS)
    
    # This fetches all devices linked to your cloud account
    devices = await device_manager.get_devices()
    
    for device in devices:
        # Option A: Using the get_alias() method
        alias = device.get_alias()
        print(f"Found device: {alias}")
        
        if alias == "Dorm Switch":
            await device.toggle()
            print("Toggled!")

async def control_plug():
    device_manager = TPLinkDeviceManager(KASA_USER, KASA_PASS)
    devices = await device_manager.get_devices()
    
    plug = None
    for d in devices:
        if d.get_alias() == DEVICE_NAME:
            plug = d
            break

    if not plug:
        print(f"Could not find device: {DEVICE_NAME}")
        return

    print(f"--- Connected to {DEVICE_NAME} ---")
    print("Commands: 'on', 'off', 'toggle', 'exit'")

    # 4. The Input Loop
    while True:
        # Standard input() works fine inside an async loop for simple scripts
        command = input("\nEnter command: ").lower().strip()

        if command == "on":
            await plug.power_on()
            print("Plug is now ON")
        elif command == "off":
            await plug.power_off()
            print("Plug is now OFF")
        elif command == "toggle":
            await plug.toggle()
            print("Plug state changed")
        elif command == "exit":
            print("Goodbye!")
            break
        else:
            print("Unknown command. Try 'on', 'off', or 'toggle'.")

if __name__ == "__main__":
    asyncio.run(control_plug())

# if __name__ == "__main__":
#     asyncio.run(main())