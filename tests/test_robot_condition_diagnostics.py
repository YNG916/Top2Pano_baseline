"""Check interventions preserve the observation and immutable source sample."""
import numpy as np
import pytest
from scripts.diagnose_robot_conditions import perturb_sample


def sample():
    camera=np.eye(4,dtype=np.float32)
    camera[:3,:3]=[[0,0,1],[1,0,0],[0,1,0]]
    return {'robot_id':1,'robot_ids':np.array([0,1,2]),
            'robot_T_wb':np.repeat(np.eye(4,dtype=np.float32)[None],3,axis=0),
            'robot_camera_heights':np.array([.8,1.,1.2],dtype=np.float32),
            'T_wc_rgb':camera,'bev_rgb':np.ones((4,4,3),dtype=np.float32)}


def test_interventions_keep_observer_camera_environment_and_source():
    original=sample()
    for mode in ('original','remove_others','move_others_right','swap_other_identities'):
        changed,_=perturb_sample(original,mode,shift_m=.75)
        observer=np.flatnonzero(changed['robot_ids']==1)
        assert len(observer)==1
        np.testing.assert_array_equal(changed['robot_T_wb'][observer[0]],original['robot_T_wb'][1])
        assert changed['robot_camera_heights'][observer[0]]==original['robot_camera_heights'][1]
        np.testing.assert_array_equal(changed['T_wc_rgb'],original['T_wc_rgb'])
        np.testing.assert_array_equal(changed['bev_rgb'],original['bev_rgb'])
        np.testing.assert_array_equal(original['robot_ids'],[0,1,2])
        np.testing.assert_array_equal(original['robot_T_wb'],np.repeat(np.eye(4)[None],3,axis=0))
    removed,_=perturb_sample(original,'remove_others')
    np.testing.assert_array_equal(removed['robot_ids'],[1])
    moved,details=perturb_sample(original,'move_others_right',.75)
    np.testing.assert_allclose(moved['robot_T_wb'][[0,2],:3,3],[[0,.75,0],[0,.75,0]])
    assert details['world_translation_m']==[0,.75,0]
    swapped,_=perturb_sample(original,'swap_other_identities')
    np.testing.assert_array_equal(swapped['robot_ids'],[2,1,0])
    np.testing.assert_array_equal(swapped['robot_T_wb'],original['robot_T_wb'])
    np.testing.assert_array_equal(swapped['robot_camera_heights'],original['robot_camera_heights'])


def test_invalid_interventions_fail():
    with pytest.raises(ValueError,match='Unknown'):perturb_sample(sample(),'bad')
    with pytest.raises(ValueError,match='finite'):perturb_sample(sample(),'original',float('nan'))
    original=sample();original['robot_ids']=np.array([0,2,3])
    with pytest.raises(ValueError,match='observer'):perturb_sample(original,'remove_others')
