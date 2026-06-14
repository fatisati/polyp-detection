export interface DemoVideo {
  label: string;
  file: string;       // filename under /demos/
  gtFile?: string;    // GT JSON filename under /demos/gt/
  thumbnail?: string; // optional thumbnail filename under /demos/
  description: string;
}

export const DEMO_VIDEOS: DemoVideo[] = [
  {
    label: "Sequence 1",
    file: "test_polyp_seq1.mp4",
    gtFile: "test_polyp_seq1_gt.json",
    description: "Positive — polyp visible throughout (53 frames)",
  },
  {
    label: "Sequence 2",
    file: "test_polyp_seq2.mp4",
    gtFile: "test_polyp_seq2_gt.json",
    description: "Intermittent — polyp appears in 135/206 frames",
  },
  {
    label: "Sequence 3",
    file: "test_polyp_seq3.mp4",
    gtFile: "test_polyp_seq3_gt.json",
    description: "Positive — polyp visible in 259/273 frames",
  },
];
