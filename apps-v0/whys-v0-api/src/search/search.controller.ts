import { Controller, Get, Query } from '@nestjs/common';
import { db } from '../db';

@Controller('search')
export class SearchController {
  @Get()
  async search(@Query('q') q: string, @Query('user_id') userId: string) {
    const embed = await embedText(q); // TODO: replace with real embeddings API call
    const res = await db.query(
      `SELECT tc.text, tc.transcript_id, 1 - (tc.embedding <=> $1) AS score
       FROM transcript_chunks tc
       JOIN transcripts t ON t.id = tc.transcript_id
       WHERE t.user_id = $2
       ORDER BY tc.embedding <=> $1
       LIMIT 20`,
      [embed, userId]
    );
    return { items: res.rows };
  }
}
async function embedText(_: string): Promise<number[]> {
  return new Array(1536).fill(0); // placeholder
}