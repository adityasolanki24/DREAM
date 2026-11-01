export class FrameOrder {
  accept(epoch: string | null, sequence: number): boolean;
}
export class MultipartJpeg {
  push(chunk: Uint8Array): { headers: Record<string, string>; body: Uint8Array }[];
}
