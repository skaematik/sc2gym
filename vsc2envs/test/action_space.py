

def test_MovementBanditsMoveToBeaconReset():
    from vsc2envs.envs.movement_bandits_beacon_reset_goal import MovementBanditsMoveToBeaconReset
    m = MovementBanditsMoveToBeaconReset()
    print('size of square ', m.screen_size / m.action_size)
    for action in range(m.action_size * m.action_size):
        print(m._translate_action_1d_to_2d(action))


def test_MovementBanditsMoveToBeaconClone():
    from vsc2envs.envs.movement_bandits_beacon_clone import MovementBanditsMoveToBeaconClone
    m = MovementBanditsMoveToBeaconClone()
    print('size of square ', m.screen_size / m.action_size)
    for action in range(m.action_size * m.action_size):
        print(m._translate_action_1d_to_2d(action))


if __name__ == "__main__":
    test_MovementBanditsMoveToBeaconClone()

