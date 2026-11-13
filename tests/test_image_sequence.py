import unittest

from detection_pipeline.image_sequence import build_image_triples, filter_camera_images


def name(stamp, number):
    return f"09_{stamp}_DSC_{number:04d}.thumb.jpg"


class ImageSequenceTests(unittest.TestCase):
    def test_empty_and_single_image(self):
        self.assertEqual(build_image_triples([]), ([], []))
        image = name("2015-01-01_120000", 1)
        self.assertEqual(build_image_triples([image]), ([], [image]))

    def test_orphans_keep_their_own_names_and_indices_are_contiguous(self):
        images = [name(stamp, i) for i, stamp in enumerate([
            "2015-01-01_100000", "2015-01-01_120000",
            "2015-01-01_120030", "2015-01-01_140000"])]
        triples, skipped = build_image_triples(images)
        self.assertEqual(skipped, [images[0], images[3]])
        self.assertEqual([t.index for t in triples], [0, 1])
        self.assertEqual((triples[0].before, triples[0].current, triples[0].after),
                         (None, images[1], images[2]))
        self.assertEqual((triples[1].before, triples[1].current, triples[1].after),
                         (images[1], images[2], None))

    def test_threshold_is_inclusive(self):
        a = name("2015-01-01_120000", 1)
        b = name("2015-01-01_120300", 2)
        self.assertEqual(len(build_image_triples([a,b])[0]), 2)
        self.assertEqual(build_image_triples([a,b], 179), ([], [a,b]))

    def test_dates_and_midnight(self):
        a = name("2015-01-01_235950", 1)
        b = name("2015-01-02_000020", 2)
        self.assertEqual(len(build_image_triples([a,b])[0]), 2)
        c = name("2015-01-03_000020", 3)
        self.assertEqual(build_image_triples([a,c]), ([], [a,c]))

    def test_reverse_times_are_not_neighbours(self):
        a = name("2015-01-01_120000", 1)
        b = name("2015-01-01_120030", 2)
        self.assertEqual(build_image_triples([b,a]), ([], [b,a]))

    def test_camera_validation_prevents_contaminating_neighbours(self):
        foreign = "16_2015-01-23_181628_DSC_0730.thumb.jpg"
        images = [foreign, name("2015-01-23_190000", 1), name("2015-01-23_190030", 2)]
        accepted, rejected = filter_camera_images(images, "DFNSMALL09")
        self.assertEqual(rejected, [foreign])
        triples, orphans = build_image_triples(accepted)
        self.assertFalse(orphans)
        self.assertTrue(all(t.before != foreign and t.after != foreign for t in triples))

    def test_generic_folders_and_numeric_prefixes(self):
        images = ["09_2015-01-01_120000_DSC_0001.thumb.jpg", "9_2015-01-01_120030_DSC_0002.thumb.jpg", "unknown.jpg"]
        self.assertEqual(filter_camera_images(images, "samples"), (images, []))
        self.assertEqual(filter_camera_images(images, "DFNSMALL09"), (images[:2], images[2:]))
        self.assertEqual(filter_camera_images(images, "DFNKIT09"), (images[:2], images[2:]))


if __name__ == "__main__":
    unittest.main()
