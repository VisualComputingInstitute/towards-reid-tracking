import numpy as np
import lib


FAKE_HEATMAP_SHAPE = (36, 64)


class FakeNeuralNewsNetwork:
    def __init__(self, dets):
        self.already_tracked_ids = [[], [], [], [], [], [], [], []]
        self.dets = dets


    def tick(self, curr_frame):
        self.curr_dets = lib.slice_all(self.dets, self.dets['GFIDs'] == curr_frame)


    def fake_camera(self, icam):
        self.curr_cam_dets = lib.slice_all(self.curr_dets, self.curr_dets['Cams'] == icam)
        self.fake_curr_cam = icam


    def embed_crop(self, crop, fake_id):
        return fake_id


    def embed_image(self, image):
        return None  # z.B. (30,60,128)


    def search_person(self, img_embs, person_emb, fake_track_id):
        id_heatmap = np.random.rand(*FAKE_HEATMAP_SHAPE)
        id_det_boxes = self.curr_cam_dets['boxes'][self.curr_cam_dets['TIDs'] == fake_track_id]
        return self._heatmap_sampling_for_dets(id_heatmap, id_det_boxes)


    def personness(self, image, known_embs):
        already_tracked_ids = self.already_tracked_ids[self.fake_curr_cam - 1]
        new_det_indices = np.where(np.logical_not(np.in1d(self.curr_cam_dets['TIDs'], already_tracked_ids)))[0]
        new_heatmaps_and_ids = []
        for each_det_idx in new_det_indices:
            new_heatmap = np.random.rand(*FAKE_HEATMAP_SHAPE)
            new_heatmap = self._heatmap_sampling_for_dets(new_heatmap, [self.curr_cam_dets['boxes'][each_det_idx]])
            new_id = self.curr_cam_dets['TIDs'][each_det_idx]
            # TODO: get correct track_id (loop heatmap, instead of function call?# )
            # TODO: get id_heatmap of that guy for init_heatmap
            already_tracked_ids.append(new_id)
            new_heatmaps_and_ids.append((new_heatmap, new_id))
        return new_heatmaps_and_ids


    def _heatmap_sampling_for_dets(self, heatmap, dets_boxes):
        H, W = heatmap.shape
        for l, t, w, h in dets_boxes:
            # score is how many times more samples than pixels in the detection box.
            score = np.random.randint(1,5)
            add_idx = np.random.multivariate_normal([l+w/2, t+h/2], [[(w/6)**2, 0], [0, (h/6)**2]], int(np.prod(heatmap.shape)*h*w*score))
            np.add.at(heatmap, [[int(np.clip(y, 0, 0.999)*H) for x,y in add_idx],
                                [int(np.clip(x, 0, 0.999)*W) for x,y in add_idx]], 1)
        return lib.softmax(heatmap)
