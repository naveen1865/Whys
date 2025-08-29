// import { Module } from '@nestjs/common';
// import { ReplyModule } from './reply/reply.module';
// import { IngestModule } from './ingest/ingest.module';

// @Module({
//   imports: [ReplyModule, IngestModule],
// })
// export class AppModule {}


import { Module } from '@nestjs/common';
import { IngestModule } from './ingest/ingest.module';
import { TimelineModule } from './timeline/timeline.module';
import { SearchModule } from './search/search.module';

@Module({ imports: [IngestModule, TimelineModule, SearchModule] })
export class AppModule {}
