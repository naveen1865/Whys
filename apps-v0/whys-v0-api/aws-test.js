// const path = require('path');
// require('dotenv').config({ path: path.join(__dirname, '.env'), override: true, debug: true });

// process.env.AWS_EC2_METADATA_DISABLED = 'true'; // avoid IMDS fallback locally

// const AWS = require('aws-sdk');
// const s3 = new AWS.S3({
//   region: process.env.S3_REGION,
//   credentials: {
//     accessKeyId: process.env.S3_ACCESS_KEY,
//     secretAccessKey: process.env.S3_SECRET_KEY,
//   },
// });

// // 1) Cheap existence/permission check
// s3.headBucket({ Bucket: process.env.S3_BUCKET }, (err) => {
//   if (err) {
//     console.error('HeadBucket error:', err);
//     return;
//   }
//   // 2) List a few keys
//   s3.listObjectsV2({ Bucket: process.env.S3_BUCKET, MaxKeys: 5 }, (err2, data) => {
//     if (err2) console.error('ListObjectsV2 error:', err2);
//     else console.log('OK. First keys:', (data.Contents || []).map(o => o.Key));
//   });
// });



// require('dotenv').config();
// const AWS = require('aws-sdk');

// const s3 = new AWS.S3({
//   region: process.env.S3_REGION,
//   accessKeyId: process.env.S3_ACCESS_KEY,
//   secretAccessKey: process.env.S3_SECRET_KEY,
// });

// const params = {
//   Bucket: process.env.S3_BUCKET,
//   Key: `test-file-${Date.now()}.wav`,
//   Expires: 300, // 1 min
// };

// s3.getSignedUrl('putObject', params, (err, url) => {
//   if (err) console.error('Error:', err);
//   else console.log('Presigned URL:', url);
// });


// check_jobs.js
const Redis = require('ioredis');

const r = new Redis(process.env.REDIS_URL || 'redis://127.0.0.1:6379');
const Q = 'whys:jobs';

(async () => {
  const len = await r.llen(Q);
  const next = await r.lindex(Q, -1);        // next job your worker will pop
  const last5 = await r.lrange(Q, -5, -1);   // last up to 5 jobs

  console.log('Length:', len);
  console.log('Next job:', next);
  console.log('Last 5:', last5);

  await r.quit();
})().catch(e => { console.error(e); process.exit(1); });
