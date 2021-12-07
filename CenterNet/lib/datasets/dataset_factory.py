
from dataset.coco import COCO



dataset_factory = {
  'coco': COCO,
  # 'pascal': PascalVOC,
  # 'kitti': KITTI,
  # 'coco_hp': COCOHP
}
_sample_factory = {
  # 'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  # 'ddd': DddDataset,
  # 'multi_pose': MultiPoseDataset
}


def get_dataset(dataset,task):
    class Dataset(dataset_factory[dataset],_sample_factory[task]):
        pass
    return Dataset