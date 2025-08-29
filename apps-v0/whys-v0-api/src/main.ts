// import { NestFactory } from '@nestjs/core';
// import { AppModule } from './app.module';

// async function bootstrap() {
//   const app = await NestFactory.create(AppModule);
//   await app.listen(5000);
//   console.log(`API running at http://localhost:5000`);
// }
// bootstrap();
import * as dotenv from 'dotenv';
dotenv.config();

import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors({ origin: true, credentials: true });
  await app.listen(Number(process.env.PORT || 5000));
  console.log(`API running at http://localhost:${process.env.PORT || 5000}`);
}
bootstrap();
