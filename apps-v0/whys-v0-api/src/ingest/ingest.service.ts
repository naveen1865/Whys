// const s3 = new S3Client({
//   region: process.env.S3_REGION,
//   credentials: {
//     accessKeyId: process.env.S3_ACCESS_KEY!,
//     secretAccessKey: process.env.S3_SECRET_KEY!
//   }
//   // NOTE: no endpoint, no forcePathStyle for AWS
// });

// import { Injectable } from '@nestjs/common';
// import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
// import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
// import Redis from 'ioredis';
// import { randomUUID } from 'crypto';

// const redis = new Redis(process.env.REDIS_URL!);

// const s3 = new S3Client({
//   region: process.env.S3_REGION,
//   credentials: {
//     accessKeyId: process.env.S3_ACCESS_KEY!,
//     secretAccessKey: process.env.S3_SECRET_KEY!,
//   },
//   // NOTE: no endpoint and no forcePathStyle for real AWS
// });

// @Injectable()
// export class IngestService {
//   async presignUpload(userId: string) {
//     const objectKey = `audio/${userId}/${randomUUID()}.m4a`;
//     const cmd = new PutObjectCommand({
//       Bucket: process.env.S3_BUCKET!,
//       Key: objectKey,
//       ContentType: 'audio/m4a',
//       // (Optional) Server-side encryption—already defaulted at bucket level, but OK to be explicit:
//       // ServerSideEncryption: 'AES256',
//     });
//     const url = await getSignedUrl(s3, cmd, {
//       expiresIn: Number(process.env.PRESIGN_EXPIRES_SECONDS || 900),
//     });
//     return { url, objectKey };
//   }

//   async enqueueJob(userId: string, s3Key: string) {
//     const job = {
//       job_id: randomUUID(),
//       user_id: userId,
//       s3_key: s3Key,
//       created_at: new Date().toISOString(),
//     };
//     await redis.lpush('whys:jobs', JSON.stringify(job));
//     return job;
//   }
// }





import { Injectable, Logger } from '@nestjs/common';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import Redis from 'ioredis';
import { randomUUID } from 'crypto';

const redis = new Redis(process.env.REDIS_URL!);

const s3 = new S3Client({
  region: process.env.S3_REGION,
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY!,
    secretAccessKey: process.env.S3_SECRET_KEY!,
  },
  // NOTE: no endpoint and no forcePathStyle for real AWS
});

@Injectable()
export class IngestService {
  private readonly logger = new Logger(IngestService.name);

  /**
   * Generate a presigned URL for uploading a user audio file.
   * @param userId - The UUID of the user
   * @returns { url, objectKey } - The presigned URL and S3 object key
   */
  async presignUpload(userId: string) {
    const objectKey = `audio/${userId}/${randomUUID()}.wav`;

    const cmd = new PutObjectCommand({
      Bucket: process.env.S3_BUCKET!,
      Key: objectKey,
      ContentType: 'audio/wav'
      // Optional: server-side encryption
      // ServerSideEncryption: 'AES256',
    });

    const url = await getSignedUrl(s3, cmd, {
      expiresIn: Number(process.env.PRESIGN_EXPIRES_SECONDS || 300), // default 15 minutes
    });

    this.logger.debug(`Generated presigned URL for user ${userId}: ${url}`);

    return { url, objectKey };
  }

  /**
   * Enqueue a job into Redis for background processing.
   * @param userId - The UUID of the user
   * @param s3Key - The S3 object key to process
   * @returns The job object stored in Redis
   */
  async enqueueJob(userId: string, s3Key: string) {
    const job = {
      job_id: randomUUID(),
      user_id: userId,
      s3_key: s3Key,
      created_at: new Date().toISOString(),
    };

    await redis.lpush('whys:jobs', JSON.stringify(job));
    this.logger.debug(`Enqueued job for user ${userId} with S3 key ${s3Key}`);

    return job;
  }

  /**
   * Convenience method: Generate presigned URL AND enqueue job.
   * @param userId - The UUID of the user
   */
  async presignAndQueue(userId: string) {
    const { url, objectKey } = await this.presignUpload(userId);
    const job = await this.enqueueJob(userId, objectKey);
    return { presign: { url, objectKey }, job };
  }
}

