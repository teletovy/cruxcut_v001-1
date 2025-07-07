"use client"

// 이 컴포넌트는 사용자가 비디오 파일을 업로드하고 AI 기반 처리 후 다운로드할 수 있는 UI를 제공함

import { useState, useCallback, useEffect } from "react"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Upload, FileVideo, CheckCircle, AlertCircle, Download, Smartphone } from "lucide-react"

const url = "https://driven-amused-tomcat.ngrok-free.app"

interface JobStatus {
  status: "queued" | "processing" | "completed" | "failed" | "converting"
  progress: number
  error?: string
  output_file?: string
}

export function FileUploadCard() {
  // 업로드할 파일 상태
  const [file, setFile] = useState<File | null>(null)
  // 업로드 중인지 여부
  const [uploading, setUploading] = useState(false)
  // 업로드 진행률 (0~100)
  const [uploadProgress, setUploadProgress] = useState(0)
  // 백엔드에서 반환한 작업 ID
  const [jobId, setJobId] = useState<string | null>(null)
  // 작업 상태 정보
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  // 에러 메시지
  const [error, setError] = useState<string | null>(null)

  // Service Worker 및 Background Mode 추가
  const [serviceWorkerReady, setServiceWorkerReady] = useState(false)
  const [backgroundMode, setBackgroundMode] = useState(false)

  // Service Worker 준비 상태 확인 및 메시지 리스너 설정
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.ready.then(() => {
        setServiceWorkerReady(true);

        // 알림 권한 요청
        if ('Notification' in window && Notification.permission === 'default') {
          Notification.requestPermission();
        }
      });

      // Service Worker 메시지 리스너
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data.type === 'JOB_STATUS_UPDATE') {
          setJobStatus(event.data.status);
        } else if (event.data.type === 'POLLING_ERROR') {
          setError(event.data.error);
        }
      });
    }
  }, []);

  // 페이지 가시성 변경 감지 및 백그라운드 폴링 위임
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (
        document.hidden &&
        jobId &&
        jobStatus &&
        (jobStatus.status === 'processing' || jobStatus.status === 'queued' || jobStatus.status === 'converting')
      ) {
        setBackgroundMode(true);
        // Service Worker에 폴링 시작 요청
        if (serviceWorkerReady) {
          navigator.serviceWorker.ready.then((registration) => {
            registration.active?.postMessage({
              type: 'START_POLLING',
              jobId,
              apiUrl: url,
            });
          });
        }
      } else if (!document.hidden && backgroundMode) {
        setBackgroundMode(false);
        // 페이지가 다시 보이면 일반 폴링 재시작
        if (jobId) {
          pollJobStatus(jobId);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [jobId, jobStatus, serviceWorkerReady, backgroundMode]);

  // 사용자가 드롭한 파일을 상태에 저장하고 초기화 처리
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0])
      setError(null)
      setJobStatus(null)
      setJobId(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [".mp4", ".mov", ".avi", ".mkv"],
    },
    maxFiles: 1,
    maxSize: 1 * 1024 * 1024 * 1024, // 1GB
  })

  // 파일을 서버로 업로드하고 처리 요청을 전송하는 함수
  const uploadAndProcess = async () => {
    if (!file) return

    setUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const xhr = new XMLHttpRequest()
      const uploadUrl = `${url}/upload`
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percent = (event.loaded / event.total) * 100
          setUploadProgress(Math.round(percent))
        }
      }

      xhr.onload = async () => {
        if (xhr.status !== 200) {
          setError(`Upload failed: ${xhr.responseText}`)
          setUploading(false)
          return
        }

        const result = JSON.parse(xhr.responseText)
        setJobId(result.job_id)
        setUploading(false)
        // 업로드 완료 후 폴링 시작
        pollJobStatus(result.job_id)
      }

      xhr.onerror = () => {
        setError("Upload failed due to network error")
        setUploading(false)
      }

      xhr.open("POST", uploadUrl)
      xhr.setRequestHeader("ngrok-skip-browser-warning", "true")
      xhr.send(formData)
    }
    catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed")
      setUploading(false)
    }
  }

  // 주기적으로 백엔드로부터 작업 상태를 확인하는 함수
  const pollJobStatus = async (id: string) => {
    // 백그라운드 모드이면 폴링 중단 (Service Worker가 처리)
    if (backgroundMode) return;

    const statusUrl = `${url}/status/${id}`
    try {
      const response = await fetch(statusUrl, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      })
      const contentType = response.headers.get("content-type") || ""

      if (!response.ok || !contentType.includes("application/json")) {
        const text = await response.text()
        throw new Error(`서버 응답이 JSON이 아님: ${text}`)
      }
      const status: JobStatus = await response.json()

      setJobStatus(status)
      // 작업이 완료되지 않았으면 계속 폴링
      if (!document.hidden && (status.status === "queued" || status.status === "processing")) {
        setTimeout(() => pollJobStatus(id), 2000)
      } else if (!document.hidden && status.status === "converting") {
        setTimeout(() => pollJobStatus(id), 5000)
      }
    } catch (err) {
      // 네트워크 오류 시 재시도 로직
      if (!document.hidden) {
        setTimeout(() => pollJobStatus(id), 5000)
      }
      setError(err instanceof Error ? err.message : "Failed to get status")
    }
  }

  // 처리된 비디오를 서버에서 다운로드하여 사용자에게 저장하게 하는 함수
  const downloadProcessedVideo = async () => {
    if (!jobId) return
    const downloadUrl = `${url}/download/${jobId}`

    try {
      const response = await fetch(downloadUrl, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      })

      if (!response.ok) {
        throw new Error("Download failed")
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.style.display = "none"
      a.href = url
      a.download = `cruxcut_processed_${jobId}.mp4`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed")
    }
  }

  // 업로드 상태를 초기화하는 함수
  const resetUpload = () => {
    setFile(null)
    setUploading(false)
    setJobStatus(null)
    setJobId(null)
    setError(null)
    setBackgroundMode(false)
  }

  // 작업 상태에 따라 사용자에게 보여줄 메시지를 반환
  const getStatusMessage = () => {
    if (!jobStatus) return ""

    switch (jobStatus.status) {
      case "queued":
        return "처리 대기 중..."
      case "processing":
        return "AI가 영상을 분석하고 편집 중입니다..."
      case "completed":
        return "처리 완료! 다운로드 버튼을 누른 후 기다려주세요."
      case "converting":
        return "영상 변환 중..."
      case "failed":
        return `처리 실패: ${jobStatus.error || "알 수 없는 오류"}`
      default:
        return ""
    }
  }

  // 주요 UI 구성 요소 렌더링 (파일 업로드, 진행률, 결과 등)
  return (
    <Card className="w-full bg-gray-900 border-gray-800">
      <CardHeader>
        <CardTitle className="text-center text-2xl">영상 업로드 및 AI 처리</CardTitle>
        {/* 백그라운드 모드 표시 */}
        {backgroundMode && (
          <div className="flex items-center justify-center gap-2 p-2 bg-blue-900/20 rounded-lg">
            <Smartphone className="h-4 w-4 text-blue-400" />
            <span className="text-sm text-blue-400">백그라운드에서 처리 중 - 알림으로 완료를 확인하세요</span>
          </div>
        )}
      </CardHeader>
      <CardContent>
        {/* 파일이 선택되지 않은 상태: 드래그 앤 드롭 영역 표시 */}
        {!file && (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive
                ? "border-purple-500 bg-purple-500/10"
                : "border-gray-700 hover:border-purple-500/50 hover:bg-gray-800/50"
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
            <p className="text-lg mb-2">
              {isDragActive ? "여기에 파일을 놓으세요" : "영상 파일을 여기에 드래그하세요"}
            </p>
            <p className="text-sm text-gray-400 mb-4">또는</p>
            <Button variant="outline" className="border-gray-700">
              파일 찾기
            </Button>
            <p className="mt-4 text-xs text-gray-500">지원 형식: MP4, MOV, AVI, MKV (최대 1GB)</p>
          </div>
        )}

        {/* 파일이 선택되고 업로드 전 상태: 파일명 및 취소 버튼 표시 */}
        {file && !jobStatus && (
          <div className="p-4 border border-gray-800 rounded-lg">
            <div className="flex items-center gap-3">
              <FileVideo className="h-8 w-8 text-purple-500" />
              <div className="flex-1">
                <p className="font-medium truncate">{file.name}</p>
                <p className="text-sm text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
              <Button variant="destructive" size="sm" onClick={resetUpload}>
                취소
              </Button>
            </div>
          </div>
        )}

        {/* 업로드 진행 중일 때 진행률 표시 */}
        {uploading && (
          <div className="space-y-4">
            <div className="flex justify-between text-sm mb-1">
              <span>업로드 중...</span>
              <span>{uploadProgress}%</span>
            </div>
            <Progress value={uploadProgress} className="h-2" />
          </div>
        )}

        {/* 작업 상태(jobStatus)가 존재할 때 진행 상황 또는 결과 표시 */}
        {jobStatus && (
          <div className="space-y-4">
            <div className="flex items-center gap-3 p-4 border border-gray-800 rounded-lg">
              {jobStatus.status === "completed" ? (
                <CheckCircle className="h-6 w-6 text-green-500" />
              ) : jobStatus.status === "failed" ? (
                <AlertCircle className="h-6 w-6 text-red-500" />
              ) : (
                <FileVideo className="h-6 w-6 text-purple-500" />
              )}
              <div className="flex-1">
                <p className="font-medium">{getStatusMessage()}</p>
                {file && <p className="text-sm text-gray-400">{file.name}</p>}
              </div>
            </div>

            {(jobStatus.status === "processing" || jobStatus.status === "queued") && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>처리 진행률</span>
                  <span>{jobStatus.progress}%</span>
                </div>
                <Progress value={jobStatus.progress} className="h-2" />
              </div>
            )}
          </div>
        )}

        {/* 오류 발생 시 오류 메시지 표시 */}
        {error && (
          <div className="flex items-center gap-3 p-4 border border-red-900/50 bg-red-900/20 rounded-lg mt-4">
            <AlertCircle className="h-6 w-6 text-red-500" />
            <p>{error}</p>
          </div>
        )}
      </CardContent>
      {/* 하단 버튼 렌더링 조건 */}
      <CardFooter className="flex justify-center gap-4">
        {file && !jobStatus && !uploading && (
          <Button
            onClick={uploadAndProcess}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
          >
            AI 처리 시작
          </Button>
        )}

        {jobStatus?.status === "completed" && (
          <Button
            onClick={downloadProcessedVideo}
            className="flex-1 bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700"
          >
            <Download className="mr-2 h-4 w-4" />
            처리된 영상 다운로드
          </Button>
        )}

        {(jobStatus?.status === "completed" || jobStatus?.status === "failed") && (
          <Button variant="outline" onClick={resetUpload}>
            새 영상 업로드
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
