export function mapPixels(cells: number[][]): {x:number;y:number;value:number}[];
export function worldToPixel(
  map: {cells: number[][]; origin_m: number[]; resolution_m: number},
  x: number,
  y: number,
): {px: number; py: number} | null;
