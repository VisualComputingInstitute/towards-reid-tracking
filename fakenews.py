import numpy as np
import lib


class FakeNeuralNewsNetwork:
    def __init__(self, dets, shape=(33, 60)):
        self.already_tracked_ids = [[], [], [], [], [], [], [], []]
        self.dets = dets
        self.shape = shape


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
        id_det_boxes = self.curr_cam_dets['boxes'][self.curr_cam_dets['TIDs'] == fake_track_id]
        return self._heatmap_sampling_for_dets(id_det_boxes)


    def personness(self, image, known_embs, return_pose=False):
        already_tracked_ids = self.already_tracked_ids[self.fake_curr_cam - 1]
        new_det_indices = np.where(np.logical_not(np.in1d(self.curr_cam_dets['TIDs'], already_tracked_ids)))[0]
        new_heatmaps_and_ids = []
        for each_det_idx in new_det_indices:
            det = self.curr_cam_dets['boxes'][each_det_idx]
            new_heatmap = self._heatmap_sampling_for_dets([det])
            if return_pose:
                new_heatmap = (new_heatmap, lib.box_center_xy(lib.box_rel2abs(det, h=self.shape[0], w=self.shape[1])))
            new_id = self.curr_cam_dets['TIDs'][each_det_idx]
            already_tracked_ids.append(new_id)
            new_heatmaps_and_ids.append((new_heatmap, new_id))
        return new_heatmaps_and_ids


    def _heatmap_sampling_for_dets(self, dets_boxes):
        heatmap = np.random.rand(*self.shape)
        for l, t, w, h in dets_boxes:
            # score is how many times more samples than pixels in the detection box.
            score = np.random.randint(1,5)
            add_idx = np.random.multivariate_normal([l+w/2, t+h/2], [[(w/6)**2, 0], [0, (h/6)**2]], int(np.prod(heatmap.shape)*h*w*score))
            np.add.at(heatmap, [[int(np.clip(y, 0, 0.999)*self.shape[0]) for x,y in add_idx],
                                [int(np.clip(x, 0, 0.999)*self.shape[1]) for x,y in add_idx]], 1)
        return lib.softmax(heatmap)


    def fix_shape(self, net_output, orig_shape, out_shape, fill_value=0):
        if net_output.shape == out_shape:
            return np.array(net_output)
        else:
            return lib.resize_map(net_output, out_shape, interp='bicubic')
