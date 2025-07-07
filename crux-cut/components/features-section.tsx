import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Crop, Scissors, Zap, Camera, Clock, Award } from "lucide-react"

export function FeaturesSection() {
  return (
    <section id="features" className="py-16 md:py-24 bg-gradient-to-b from-black to-gray-900">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">CruxCut의 특별한 기능</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            고정된 카메라로 촬영한 영상을 AI가 분석하여 클라이머를 중심으로 한 역동적인 영상으로 변환합니다.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <Crop className="h-10 w-10 text-purple-500 mb-2" />
              <CardTitle>자동 크롭 및 추적</CardTitle>
              <CardDescription>
                AI가 클라이머를 자동으로 인식하고 추적하여 항상 프레임 중앙에 위치시킵니다.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400">
                복잡한 움직임도 놓치지 않고 부드럽게 따라가는 스마트 트래킹 기술로 마치 전문 촬영팀이 촬영한 것 같은
                결과물을 얻을 수 있습니다.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <Scissors className="h-10 w-10 text-purple-500 mb-2" />
              <CardTitle>하이라이트 자동 편집</CardTitle>
              <CardDescription>중요한 순간을 자동으로 감지하여 하이라이트 영상을 생성합니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400">
                크럭스 구간, 다이나믹한 움직임, 성공 순간 등을 AI가 자동으로 인식하여 최적의 하이라이트 영상을 만들어
                드립니다.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <Zap className="h-10 w-10 text-purple-500 mb-2" />
              <CardTitle>원클릭 처리</CardTitle>
              <CardDescription>복잡한 설정 없이 한 번의 클릭으로 모든 과정이 자동화됩니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400">
                영상을 업로드하고 '자동 편집 시작' 버튼만 누르면 AI가 모든 작업을 처리합니다. 편집 경험이 없어도 전문가
                수준의 결과물을 얻을 수 있습니다.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <Camera className="h-10 w-10 text-purple-500 mb-2" />
              <CardTitle>다양한 카메라 앵글</CardTitle>
              <CardDescription>하나의 고정 영상에서 다양한 카메라 앵글의 영상을 생성합니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400">
                와이드샷, 클로즈업, 팔로우샷 등 다양한 앵글의 영상을 하나의 소스에서 생성하여 마치 여러 대의 카메라로
                촬영한 것 같은 효과를 제공합니다.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <Clock className="h-10 w-10 text-purple-500 mb-2" />
              <CardTitle>빠른 처리 속도</CardTitle>
              <CardDescription>고성능 AI 엔진으로 빠른 시간 내에 결과물을 제공합니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400">
                일반적인 10분 길이의 영상을 약 5분 이내에 처리 완료하여 빠르게 결과물을 확인하고 공유할 수 있습니다.
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <Award className="h-10 w-10 text-purple-500 mb-2" />
              <CardTitle>전문가급 품질</CardTitle>
              <CardDescription>전문 영상 편집자가 만든 것 같은 고품질 결과물을 제공합니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-gray-400">
                수천 시간의 클라이밍 영상으로 학습된 AI가 움직임의 역동성과 중요한 순간을 정확히 포착하여 전문가급
                품질의 영상을 만들어 드립니다.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
