"""
Tests the SokobanOracleWrapper to ensure observation shapes, 
oracle interceptions, and reward penalties work as expected.
"""

from env_wrapper import SokobanOracleWrapper
import matplotlib.pyplot as plt

def main():
    print("Initializing Wrapper...")
    # We set a high oracle cost so we can clearly see it being subtracted
    env = SokobanOracleWrapper(env_id='Sokoban-small-v0', oracle_cost=0.5)
    
    # 1. Test Reset & Observation Shape
    obs, info = env.reset(seed=42)
    print("\n--- RESET ---")
    print(f"Observation shape: {obs.shape} (Expected: (84, 84, 3))")
    print(f"Observation dtype: {obs.dtype} (Expected: uint8)")
    
    # Show the downsampled 84x84 image just to prove it looks right
    plt.imshow(obs)
    plt.title("Downsampled 84x84 Sokoban")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # 2. Test a Normal Action (Action 1: Push Up)
    print("\n--- NORMAL ACTION (1: Push Up) ---")
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"Reward: {reward}")
    print(f"Guided: {info.get('guided')}")
    print(f"Oracle Action Taken: {info.get('oracle_action')}")

    # 3. Test the Oracle Action (Action 9)
    print("\n--- ORACLE ACTION (9: Query Oracle) ---")
    obs, reward, terminated, truncated, info = env.step(9)
    print(f"Reward: {reward} (Should include the -0.5 penalty!)")
    print(f"Guided: {info.get('guided')} (Expected: True)")
    print(f"Oracle Action Taken: {info.get('oracle_action')} (Expected: 1, 2, 3, or 4)")
    
    if info.get('fatal_deadlock'):
        print("\n[!] FATAL DEADLOCK TRIGGERED! The oracle returned 0.")

    env.close()
    print("\nWrapper Test Complete!")

if __name__ == "__main__":
    main()