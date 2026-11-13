import unittest

from object_detection.val.publication_metrics import match_boxes, score_samples, source_frame_id


def sample(identity='tile-1', truth=None, predictions=None):
    return dict(sample_id=identity,source_frame='source',width=400,height=400,
                ground_truth=truth or [],predictions=[dict(box=b) for b in (predictions or [])])


class PublicationMetricTests(unittest.TestCase):
    def test_duplicate_predictions_are_false_positives(self):
        truth=[dict(observation_id='obs',box=[10,10,30,30])]
        s,_,_=score_samples([sample(truth=truth,predictions=[[10,10,30,30]]*2)])
        self.assertEqual(s['true_positive_boxes'],1)
        self.assertEqual(s['false_positive_boxes'],1)
        self.assertEqual(s['box_precision'],0.5)

    def test_maximum_cardinality_requires_reassignment(self):
        # First prediction can match either target, second can match only target 0.
        self.assertEqual(len(match_boxes([[0,0,10,10],[8,0,18,10]],
                                        [[0,0,18,10],[0,0,7,10]])),2)

    def test_multi_observation_and_overlapping_tiles(self):
        truth=[dict(observation_id='a',box=[10,10,30,30]),dict(observation_id='b',box=[100,100,120,120])]
        s,_,_=score_samples([sample(truth=truth,predictions=[[10,10,30,30]]),
                            sample('tile-2',truth=truth,predictions=[[100,100,120,120]])])
        self.assertEqual(s['observations'],2)
        self.assertEqual(s['recovered_observations'],2)
        self.assertEqual(s['false_negative_boxes'],2)

    def test_negative_only_and_empty_predictions(self):
        s,_,_=score_samples([sample(predictions=[[1,1,20,20]])])
        self.assertIsNone(s['observation_recovery'])
        self.assertEqual(s['false_positive_boxes'],1)
        s,_,_=score_samples([sample(truth=[dict(observation_id='a',box=[1,1,20,20])])])
        self.assertEqual(s['observation_recovery'],0)
        self.assertIsNone(s['box_precision'])

    def test_edge_clipping_and_padding_prediction(self):
        truth=[dict(observation_id='a',box=[-5,0,15,20])]
        s,_,_=score_samples([sample(truth=truth,predictions=[[0,0,15,20],[-5,5,-1,20]])],metric='iou',threshold=1)
        self.assertEqual(s['true_positive_boxes'],1)
        self.assertEqual(s['false_positive_boxes'],1)

    def test_exclusions_and_duplicate_ids(self):
        row=sample();row.update(eligible=False,exclusion_reason='unresolved multi-streak annotation')
        s,_,excluded=score_samples([row])
        self.assertEqual(s['evaluated_tiles'],0)
        self.assertEqual(len(excluded),1)
        with self.assertRaises(ValueError):score_samples([sample(),sample()])

    def test_source_ids_preserve_special_camera_sequences(self):
        for name in ['07_2015-12-07_193959_S_DSC_0837',
                     '36_2015-09-24_164558_00155288_DSC_7020',
                     '03_2020-08-20_070659_K_DSC_6995']:
            self.assertEqual(source_frame_id(name+'_negative_12.jpg'),name)


if __name__=='__main__':unittest.main()
