import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Check } from "lucide-react"

export function PricingSection() {
  return (
    <section id="pricing" className="py-16 md:py-24 bg-gradient-to-b from-gray-900 to-black">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">심플한 가격 정책</h2>
          <p className="text-gray-400 max-w-2xl mx-auto">
            필요에 맞는 플랜을 선택하고 지금 바로 CruxCut을 경험해보세요.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-xl">무료 체험</CardTitle>
              <div className="mt-4">
                <span className="text-4xl font-bold">₩0</span>
                <span className="text-gray-400 ml-2">/ 월</span>
              </div>
              <CardDescription className="mt-2">CruxCut의 기본 기능을 체험해보세요.</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>월 3개 영상 처리</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>최대 5분 길이 영상</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>기본 자동 크롭 및 추적</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>720p 해상도</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button variant="outline" className="w-full border-gray-700">
                무료로 시작하기
              </Button>
            </CardFooter>
          </Card>

          <Card className="bg-gray-900 border-gray-800 relative md:scale-110 z-10 shadow-xl">
            <div className="absolute -top-4 left-0 right-0 flex justify-center">
              <span className="bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-medium px-4 py-1 rounded-full">
                인기 플랜
              </span>
            </div>
            <CardHeader>
              <CardTitle className="text-xl">프로</CardTitle>
              <div className="mt-4">
                <span className="text-4xl font-bold">₩19,900</span>
                <span className="text-gray-400 ml-2">/ 월</span>
              </div>
              <CardDescription className="mt-2">개인 클라이머에게 최적화된 플랜입니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>월 30개 영상 처리</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>최대 30분 길이 영상</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>고급 자동 크롭 및 추적</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>하이라이트 자동 편집</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>1080p 해상도</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>우선 처리</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700">
                프로 시작하기
              </Button>
            </CardFooter>
          </Card>

          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="text-xl">비즈니스</CardTitle>
              <div className="mt-4">
                <span className="text-4xl font-bold">₩49,900</span>
                <span className="text-gray-400 ml-2">/ 월</span>
              </div>
              <CardDescription className="mt-2">클라이밍 센터와 팀을 위한 플랜입니다.</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>월 100개 영상 처리</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>무제한 길이 영상</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>최고급 자동 크롭 및 추적</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>하이라이트 자동 편집</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>4K 해상도</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>최우선 처리</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>브랜딩 및 워터마크 옵션</span>
                </li>
                <li className="flex items-start">
                  <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5" />
                  <span>전용 고객 지원</span>
                </li>
              </ul>
            </CardContent>
            <CardFooter>
              <Button variant="outline" className="w-full border-gray-700">
                비즈니스 시작하기
              </Button>
            </CardFooter>
          </Card>
        </div>
      </div>
    </section>
  )
}
