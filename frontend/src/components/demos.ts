export interface DemoVideo {
  label: string;
  file: string;      // filename under /demos/
  thumbnail?: string; // optional thumbnail filename under /demos/
  description: string;
}

export const DEMO_VIDEOS: DemoVideo[] = [
  {
    label: "Sequence 1",
    file: "test_polyp_seq1.mp4",
    description: "Positive — polyp visible mid-sequence",
  },
];
